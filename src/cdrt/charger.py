"""Phase 2: onboard-charger power stage, current loop, CC-CV logic and EIS injection.

System framing (all values are ASSUMED design values, not taken from a datasheet):
  * 96s3p pack of LG M50 cells (Chen2020 parameter set), nominal ~350 V, 15 Ah.
  * Second (DC-DC) stage of an on-board charger: bidirectional synchronous buck
    (half-bridge) from a 450 V DC link, f_sw = 100 kHz, L = 470 uH, output film
    capacitor C_o = 10 uF (ESR 10 mOhm), dead time 150 ns.
  * Digital average-current-mode control of the inductor current, sampled once per
    switching period at the PWM midpoint (ripple-free sample), 1-period computation delay.
  * EIS: the perturbation is added to the current reference. The charger measures the
    pack current; the BMS front-end (AFE) measures every cell voltage.
Each parallel group of 3 cells carries I_pack/3; the impedance of one cell is diagnosed.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from numba import njit

N_SERIES, N_PAR = 96, 3


@dataclass
class ConverterDesign:
    V_dc: float = 450.0
    f_sw: float = 100e3
    L: float = 470e-6
    R_L: float = 30e-3
    C_o: float = 10e-6
    ESR_C: float = 10e-3
    t_dead: float = 150e-9
    f_c: float = 5e3          # target current-loop crossover
    pm_zero_ratio: float = 10  # PI zero placed f_c/10
    f_sense: float = 50e3     # current-sensor bandwidth (1st order)
    sense_location: str = "inductor"  # "inductor" (common) or "battery"

    @property
    def Ts(self):
        return 1.0 / self.f_sw

    @property
    def Kp(self):  # crossover where |Kp * Vdc / (sL)| = 1
        return 2 * np.pi * self.f_c * self.L / self.V_dc

    @property
    def Ki(self):  # integral gain [1/s]-scaled such that zero at f_c/ratio
        return self.Kp * 2 * np.pi * self.f_c / self.pm_zero_ratio

    def ripple_pp(self, V_bat):
        D = V_bat / self.V_dc
        return V_bat * (1 - D) / (self.L * self.f_sw)


def loop_transfer(des: ConverterDesign, f, Z_pack):
    """Analytical small-signal model.
    Returns dict with loop gain T_loop, closed-loop i_L/i_ref, battery-current transfer
    i_bat/i_ref, and the sensing ratio i_bat/i_sensed (=1 if sensing at the battery)."""
    s = 2j * np.pi * np.asarray(f)
    Zc = 1 / (s * des.C_o) + des.ESR_C
    Zo = Zc * Z_pack / (Zc + Z_pack)
    G_id = des.V_dc / (s * des.L + des.R_L + Zo)          # duty -> inductor current
    k_b = Zc / (Zc + Z_pack)                               # i_bat / i_L
    Ts = des.Ts
    delay = np.exp(-1.5 * s * Ts)                          # 1 Ts computation + 0.5 Ts ZOH/PWM
    C_pi = des.Kp + des.Ki / s
    H_s = 1 / (1 + s / (2 * np.pi * des.f_sense))
    if des.sense_location == "inductor":
        G = G_id
        sensed_to_bat = k_b
    else:
        G = G_id * k_b
        sensed_to_bat = np.ones_like(k_b)
    T_loop = C_pi * delay * G * H_s
    T_cl = C_pi * delay * G / (1 + T_loop)                 # controlled current / reference
    i_bat_ref = T_cl * sensed_to_bat if des.sense_location == "inductor" else T_cl
    return dict(T_loop=T_loop, T_cl=T_cl, i_bat_over_ref=i_bat_ref,
                bat_over_sensed=sensed_to_bat, H_s=H_s)


def stability_margins(des, Z_pack_fn, f=np.logspace(1, np.log10(5e4), 4000)):
    T = loop_transfer(des, f, Z_pack_fn(f))["T_loop"]
    mag = np.abs(T)
    i = np.where(np.diff(np.sign(mag - 1)))[0]
    fc = f[i[0]] if len(i) else np.nan
    pm = 180 + np.degrees(np.angle(T[i[0]])) if len(i) else np.nan
    ph = np.unwrap(np.angle(T))
    j = np.where(np.diff(np.sign(ph + np.pi)))[0]
    gm = -20 * np.log10(mag[j[0]]) if len(j) else np.inf
    return dict(f_cross=float(fc), phase_margin_deg=float(pm), gain_margin_dB=float(gm))


# --------------------------------------------------------------------------------------
# Cycle-averaged time-domain simulation with dead time, ADC quantisation and a pack ECM
# --------------------------------------------------------------------------------------
@njit(cache=True)
def _sim_avg(n_cyc, sub, Ts, Vdc, L, RL, Co, ESR, td, Kp, Ki, iref, ocv_pack,
             R0, Rk, tauk, adc_lsb, i_noise, seed, sense_inductor, f_sense):
    np.random.seed(seed)
    dt = Ts / sub
    iL = iref[0]; vc = ocv_pack + R0 * iL
    nk = Rk.shape[0]
    vk = np.zeros(nk)
    for k in range(nk):
        vk[k] = Rk[k] * iL
    integ = 0.0
    d_prev = (ocv_pack + R0 * iL) / Vdc
    d_cmd = d_prev
    i_sense_f = iL
    a_s = np.exp(-2 * np.pi * f_sense * Ts)
    out_iL = np.zeros(n_cyc); out_ib = np.zeros(n_cyc); out_vb = np.zeros(n_cyc)
    out_d = np.zeros(n_cyc)
    ib = iL
    for n in range(n_cyc):
        # ---- sampling at the PWM midpoint (ripple-free average current) ----
        meas = sense_inductor * iL + (1 - sense_inductor) * ib
        i_sense_f = a_s * i_sense_f + (1 - a_s) * meas
        q = np.round((i_sense_f + i_noise * np.random.randn()) / adc_lsb) * adc_lsb
        e = iref[n] - q
        integ += Ki * Ts * e
        # duty computed now, applied next period (1-cycle delay), feed-forward of v_bat
        d_new = Kp * e + integ + vc / Vdc
        if d_new > 0.98: d_new = 0.98
        if d_new < 0.02: d_new = 0.02
        d_apply = d_cmd
        d_cmd = d_new
        # ---- dead-time volt-second error, using peak/valley currents ----
        vbat_est = vc
        dI = vc * (1 - d_apply) / (L / Ts) if L > 0 else 0.0
        ipk = iL + dI / 2; ival = iL - dI / 2
        sp = 1.0 if ipk > 0 else -1.0
        sv = 1.0 if ival > 0 else -1.0
        v_dead = -0.5 * Vdc * td / Ts * (sp + sv)
        vsw = d_apply * Vdc + v_dead
        for m in range(sub):
            # battery: v_b = ocv + R0 ib + sum vk ; capacitor branch with ESR
            vb_nok = ocv_pack
            for k in range(nk):
                vb_nok += vk[k]
            # solve node: vc_node = vc + ESR*ic ; ib = (vnode - vb_nok)/R0 ; ic = iL - ib
            vnode = (vc / ESR + vb_nok / R0 + iL) / (1 / ESR + 1 / R0)
            ib = (vnode - vb_nok) / R0
            ic = iL - ib
            iL += dt * (vsw - RL * iL - vnode) / L
            vc += dt * ic / Co
            for k in range(nk):
                vk[k] += dt * (Rk[k] * ib - vk[k]) / tauk[k]
        out_iL[n] = iL; out_ib[n] = ib; out_vb[n] = vnode; out_d[n] = d_apply
    return out_iL, out_ib, out_vb, out_d


def simulate_cycle_averaged(des: ConverterDesign, iref, ocv_pack, ecm, sub=8,
                            adc_bits=12, adc_fs=50.0, i_noise=0.0, seed=0):
    """ecm: dict(R0, Rk[], tauk[]) for the PACK. Returns per-switching-period arrays."""
    lsb = 2 * adc_fs / 2 ** adc_bits
    tk = np.maximum(np.asarray(ecm["tauk"], float), 4 * des.Ts / sub)
    return _sim_avg(len(iref), sub, des.Ts, des.V_dc, des.L, des.R_L, des.C_o, des.ESR_C,
                    des.t_dead, des.Kp, des.Ki, np.asarray(iref, float), float(ocv_pack),
                    float(ecm["R0"]), np.asarray(ecm["Rk"], float), tk, lsb, i_noise, seed,
                    1.0 if des.sense_location == "inductor" else 0.0, des.f_sense)


@njit(cache=True)
def _sim_switching(n_cyc, ppc, Ts, Vdc, L, RL, Co, ESR, td, duty, vbat, Rb):
    """Sub-cycle switching simulation of the half-bridge with dead time (fixed duty),
    battery as Thevenin source vbat + Rb. Used only for the illustration figure."""
    dt = Ts / ppc
    n = n_cyc * ppc
    t = np.zeros(n); iL = np.zeros(n); vsw = np.zeros(n)
    i = vbat * 0 + 0.0
    vc = vbat
    for k in range(n):
        tc = (k % ppc) * dt
        ton = duty * Ts
        if tc < td:                     # dead time after low-side off: diode decides
            v = Vdc if i < 0 else 0.0
        elif tc < ton:
            v = Vdc
        elif tc < ton + td:             # dead time after high-side off
            v = Vdc if i < 0 else 0.0
        else:
            v = 0.0
        vnode = (vc / ESR + vbat / Rb + i) / (1 / ESR + 1 / Rb)
        ib = (vnode - vbat) / Rb
        i += dt * (v - RL * i - vnode) / L
        vc += dt * (i - ib) / Co
        t[k] = k * dt; iL[k] = i; vsw[k] = v
    return t, iL, vsw


# --------------------------------------------------------------------------------------
# Battery (Butler-Volmer) nonlinearity estimate
# --------------------------------------------------------------------------------------
def bv_relative_distortion(eta_amp_V, T=298.15):
    """Relative third-harmonic distortion of the faradaic current for a sinusoidal
    overpotential of amplitude eta (symmetric BV, alpha = 0.5): x^2/24, x = F eta / (2RT)."""
    F, R = 96485.0, 8.314
    x = F * np.asarray(eta_amp_V) / (2 * R * T)
    return x ** 2 / 24
