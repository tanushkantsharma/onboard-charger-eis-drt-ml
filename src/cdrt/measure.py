"""Phase 2: fast frequency-domain model of an onboard-charger EIS measurement.

Given the true (clean) impedance of a cell on a SOC grid, return what the charger would
estimate, line by line. The model is a deterministic + stochastic error budget that is
validated against a sample-level time-domain simulation in scripts/phase2_charger.py.

Excitation: one multisine per frequency decade (measured high -> low frequency). In a band
starting at f_lo the multisine period is T0 = 2/f_lo and the excited harmonics of
f0 = f_lo/2 are m = 2,3,4,5,7,9,12,16 (~8 lines per decade, roughly log-spaced).
Schroeder phases; the per-line amplitude follows from the requested PEAK current.
One extra period is discarded as transient; P periods are averaged.

Error sources (per excited line k):
  1. current-loop transfer i_bat/i_ref  -> reduces the injected current at high f (SNR)
  2. current-sensing location            -> i_bat/i_sensed = Zc/(Zc+Zpack) bias (inductor sensing)
  3. voltage / current channel noise     -> white + 1/f, incl. ADC quantisation and
                                            residual switching-ripple aliasing
  4. anti-alias filter mismatch + inter-channel sampling skew -> multiplicative error
  5. SOC drift during the test           -> Z(f, SOC(t)) averaged over the record +
                                            OCV-ramp leakage (with optional linear detrend)
  6. non-linearity                       -> Butler-Volmer (battery) and dead-time
                                            zero-crossing distortion (converter), as noise
  7. BMS SOC error                       -> SOC fed to the classifier is noisy
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace, asdict
import numpy as np
from scipy.interpolate import CubicSpline
from .charger import ConverterDesign, loop_transfer, N_SERIES, N_PAR, bv_relative_distortion

HARM = np.array([2, 3, 4, 5, 7, 9, 12, 16])


@dataclass
class MeasConfig:
    amp_pk_cell: float = 0.5          # peak of the injected multisine, per cell [A]  (~C/10)
    f_min: float = 0.01               # lowest band start [Hz]
    f_max: float = 1000.0             # highest excited line [Hz]
    n_periods: int = 2                # averaged periods per band
    i_dc_cell: float = 0.0            # DC current during the test, per cell [A] (+ = charging)
    fs: float = 25e3                  # sampling rate of both channels [Hz]
    # voltage channel (BMS AFE, per cell)
    sigma_v: float = 100e-6           # white noise per sample [V rms]
    v_bits: int = 16; v_range: float = 5.0
    # current channel (charger, pack current)
    sigma_i_pack: float = 10e-3       # white noise per sample [A rms]
    i_bits: int = 12; i_range_pack: float = 100.0     # full scale span [A]
    ripple_alias_pack: float = 0.01   # residual aliased switching ripple [A rms]
    f_flicker: float = 1.0            # 1/f corner of both channels [Hz]
    noise_scale: float = 1.0          # multiplies all noise std (sweep knob)
    # channel mismatch
    aa_fc: float = 10e3               # anti-alias cut-off (2nd-order Butterworth)
    aa_tol: float = 0.05              # 1-sigma relative cut-off tolerance per channel (residual after calibration)
    skew: float = 5e-6                # residual inter-channel sampling skew, common to all units [s]
    skew_unit_std: float = 0.0        # extra skew drawn independently for every charger unit, 1 sd [s]
    # drift / non-linearity
    detrend: bool = True
    relax_amp: float = 0.0            # OCV relaxation amplitude at test start [V]
    relax_tau: float = 1200.0         # its time constant [s]
    include_nonlinear: bool = True
    d_conv_crossing: float = 0.01     # relative converter distortion when current crosses zero
    soc_err_std: float = 0.02         # BMS SOC error fed to the classifier
    converter: ConverterDesign = field(default_factory=ConverterDesign)

    def as_row(self):
        d = asdict(self); d.pop("converter")
        d["sense_location"] = self.converter.sense_location
        return d


def band_plan(cfg: MeasConfig):
    """List of (f_lo, T0, line frequencies) bands, measured from high to low frequency."""
    bands = []
    k0 = int(np.floor(np.log10(cfg.f_min) + 1e-9))
    k1 = int(np.ceil(np.log10(cfg.f_max) - 1e-9))
    for k in range(k0, k1):
        f_lo = 10.0 ** k
        if f_lo < cfg.f_min * (1 - 1e-9):
            continue
        lines = HARM * f_lo / 2
        lines = lines[lines <= cfg.f_max * (1 + 1e-9)]
        if len(lines):
            bands.append((f_lo, 2.0 / f_lo, lines))
    bands.sort(key=lambda b: -b[0])
    return bands


def test_duration(cfg: MeasConfig):
    return sum((cfg.n_periods + 1) * T0 for _, T0, _ in band_plan(cfg))


_UNIT_PEAK = None


def unit_peak():
    """Peak of a unit-amplitude Schroeder multisine on HARM (computed numerically)."""
    global _UNIT_PEAK
    if _UNIT_PEAK is None:
        n = len(HARM)
        ph = -np.pi * np.arange(1, n + 1) * np.arange(n) / n     # Schroeder phases
        t = np.linspace(0, 1, 20000, endpoint=False)
        x = np.cos(2 * np.pi * HARM[:, None] * t[None, :] + ph[:, None]).sum(0)
        _UNIT_PEAK = float(np.abs(x).max())
    return _UNIT_PEAK


def _interp_logf(freqs, Zsoc, fq):
    """Cubic interpolation of Re/Im vs log10 f. Zsoc: [n_soc, n_f] -> [n_soc, len(fq)]."""
    lf, lq = np.log10(freqs), np.log10(fq)
    re = CubicSpline(lf, Zsoc.real, axis=1)(lq)
    im = CubicSpline(lf, Zsoc.imag, axis=1)(lq)
    return re + 1j * im


def _butter2(f, fc):
    s = 1j * f / fc
    return 1.0 / (s ** 2 + np.sqrt(2) * s + 1)


def measure(Zsoc, freqs, soc_grid, ocv, Q_ah, soc0, cfg: MeasConfig, rng, unit=None):
    """Simulate one charger EIS test.
    Zsoc [n_soc, n_f]: clean impedance; ocv [n_soc]; Q_ah: cell capacity; soc0: start SOC.
    unit: dict with the per-charger random channel tolerances (drawn once per charger).
    Returns dict(f, Z, sigma, Z_true_avg, duration_s, dsoc, soc_reported)."""
    soc_grid = np.asarray(soc_grid, float)
    if unit is None:
        unit = draw_unit(cfg, rng)
    bands = band_plan(cfg)
    a_line = cfg.amp_pk_cell / unit_peak()
    ns = cfg.noise_scale
    # per-sample white-noise variances (cell referred)
    lsb_v = cfg.v_range / 2 ** cfg.v_bits
    lsb_i = cfg.i_range_pack / 2 ** cfg.i_bits
    var_v = (cfg.sigma_v ** 2 + lsb_v ** 2 / 12) * ns ** 2
    var_i = (cfg.sigma_i_pack ** 2 + lsb_i ** 2 / 12 + cfg.ripple_alias_pack ** 2) / N_PAR ** 2 * ns ** 2
    Sv_w, Si_w = 2 * var_v / cfg.fs, 2 * var_i / cfg.fs          # one-sided white PSD
    rate = cfg.i_dc_cell / (3600.0 * Q_ah)                         # dSOC/dt
    ocv_fn = lambda z: np.interp(z, soc_grid, ocv)
    t_start = 0.0
    fs_out, Zm, sig_out, Zt_out = [], [], [], []
    # dead-time zero crossing: does the pack current cross zero during the test?
    ripple_pp_cell = cfg.converter.ripple_pp(3.7 * N_SERIES) / N_PAR
    crossing = abs(cfg.i_dc_cell) < cfg.amp_pk_cell + ripple_pp_cell / 2
    for f_lo, T0, lines in bands:
        P = cfg.n_periods
        T_rec = P * T0
        t0 = t_start + T0                      # first period = transient, discarded
        # ---- impedance averaged over the record (SOC drift) ----
        tt = np.linspace(t0, t0 + T_rec, 41)
        soc_t = np.clip(soc0 + rate * tt, soc_grid[0], soc_grid[-1])
        Zl = _interp_logf(freqs, Zsoc, lines)               # [n_soc, n_lines]
        Zt = np.array([np.interp(soc_t, soc_grid, Zl[:, j].real).mean()
                       + 1j * np.interp(soc_t, soc_grid, Zl[:, j].imag).mean() for j in range(len(lines))])
        Z0 = np.array([np.interp(soc0, soc_grid, Zl[:, j].real) + 1j * np.interp(soc0, soc_grid, Zl[:, j].imag)
                       for j in range(len(lines))])
        # ---- injected battery current per line ----
        lt = loop_transfer(cfg.converter, lines, Zt * N_SERIES / N_PAR)
        I_bat = a_line * lt["i_bat_over_ref"]                # complex amplitude, battery
        I_sensed = I_bat / lt["bat_over_sensed"]             # what the sensor sees
        V = Zt * I_bat
        # ---- drift voltage leakage (OCV ramp + relaxation), per period then averaged ----
        n_t = 64 * HARM.max() * P
        t = t0 + np.arange(n_t) * (T_rec / n_t)
        vd = ocv_fn(np.clip(soc0 + rate * t, soc_grid[0], soc_grid[-1]))
        vd = vd + cfg.relax_amp * np.exp(-t / cfg.relax_tau)
        # per-period estimator: offset (+ linear trend if detrend) fitted jointly with the lines,
        # which is equivalent to projecting the drift out with the same basis
        n0 = n_t // P
        E = np.exp(-2j * np.pi * lines[None, :] * (t[:n0, None] - t0))
        D = np.zeros((P, len(lines)), complex)
        for p in range(P):
            seg = vd[p * n0:(p + 1) * n0]
            tp = t[p * n0:(p + 1) * n0]
            A = np.vstack([np.ones_like(tp), tp - tp.mean()]).T if cfg.detrend else np.ones((n0, 1))
            seg = seg - A @ np.linalg.lstsq(A, seg, rcond=None)[0]
            D[p] = 2 / n0 * (seg[:, None] * E).sum(0)
        # ---- noise per period (white + 1/f), per component variance = S/T0 ----
        Sv = Sv_w * (1 + cfg.f_flicker / lines)
        Si = Si_w * (1 + cfg.f_flicker / lines)
        sd_v = np.sqrt(Sv / T0)                              # std of Re (and of Im) per period
        sd_i = np.sqrt(Si / T0)
        if cfg.include_nonlinear:
            # overpotential amplitude across the interface ~ peak current x ~30 mOhm
            # (charge-transfer + film resistance of this cell, Phase 1)
            eta = cfg.amp_pk_cell * 0.03
            d_bv = float(bv_relative_distortion(eta))
            sd_v = np.sqrt(sd_v ** 2 + (d_bv * np.abs(V)) ** 2 / 2)
            if crossing:
                sd_i = np.sqrt(sd_i ** 2 + (cfg.d_conv_crossing * np.abs(I_bat)) ** 2 / 2)
        cn = lambda sd: rng.normal(0, 1, (P, len(lines))) * sd + 1j * rng.normal(0, 1, (P, len(lines))) * sd
        Vp = V[None, :] + D + cn(sd_v)
        Ip = I_sensed[None, :] + cn(sd_i)
        # ---- channel transfer: AA filters with per-unit tolerance, residual skew ----
        Hv = _butter2(lines, cfg.aa_fc * (1 + unit["dv"])) / _butter2(lines, cfg.aa_fc)
        Hi = _butter2(lines, cfg.aa_fc * (1 + unit["di"])) / _butter2(lines, cfg.aa_fc) \
            * np.exp(-2j * np.pi * lines * (cfg.skew + unit.get("skew", 0.0)))
        Zp = (Hv * Vp) / (Hi * Ip)                           # per-period estimates
        Zhat = Zp.mean(0)
        if P >= 2:
            sig = np.sqrt((Zp.real.var(0, ddof=1) + Zp.imag.var(0, ddof=1)) / 2 / P)
        else:   # noise-floor estimate from ~10 neighbouring non-excited bins
            sig_th = np.abs(Zt) * np.sqrt(sd_v ** 2 / np.abs(V) ** 2 + sd_i ** 2 / np.abs(I_bat) ** 2)
            sig = sig_th * np.sqrt(rng.chisquare(20, len(lines)) / 20)
        fs_out.append(lines); Zm.append(Zhat); sig_out.append(np.maximum(sig, 1e-9)); Zt_out.append(Zt)
        t_start += (P + 1) * T0
    order = np.argsort(np.concatenate(fs_out))
    cat = lambda L: np.concatenate(L)[order]
    dsoc = rate * t_start
    return dict(f=cat(fs_out), Z=cat(Zm), sigma=cat(sig_out), Z_true_avg=cat(Zt_out),
                duration_s=t_start, dsoc=dsoc,
                soc_reported=float(soc0 + rng.normal(0, cfg.soc_err_std)))


def draw_unit(cfg: MeasConfig, rng):
    """Per-charger (unit-to-unit) channel tolerances, fixed for all tests of that unit."""
    u = dict(dv=float(rng.normal(0, cfg.aa_tol)), di=float(rng.normal(0, cfg.aa_tol)))
    if cfg.skew_unit_std > 0:   # drawn only when used, so earlier random streams are unchanged
        u["skew"] = float(rng.normal(0, cfg.skew_unit_std))
    return u
