"""Phase 2: charger model, loop verification, dead-time distortion, CC-CV demo,
validation of the fast measurement model, and 'measured vs clean' spectra.

Outputs: figures/fig2*.png, results/phase2_*.csv
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dataclasses import replace
from cdrt import config as C, drt, charger as ch, measure as ms
from cdrt.plotstyle import plt, save, CAT, INK2, SEQ

RNG = np.random.default_rng(C.SEED + 2)
D = np.load(C.DATA / "clean_population.npz")
META = pd.read_csv(C.DATA / "clean_population_meta.csv")
SOC = np.round(D["soc_grid"], 2)
F = D["freqs"]
SCALE = ch.N_SERIES / ch.N_PAR          # pack impedance = 32 x cell impedance


def zfun(Zcell):
    lf = np.log10(F)
    return lambda ff: SCALE * (np.interp(np.log10(ff), lf, Zcell.real) + 1j * np.interp(np.log10(ff), lf, Zcell.imag))


def pack_ecm(Zcell):
    """RC-ladder equivalent of the pack from the cell DRT (used by the time-domain sims)."""
    B = drt.DRTBasis(F)
    x = B.fit(Zcell, sigma=0.005 * np.abs(Zcell), lam=1e4)
    Rk = x[3:] * B.area * SCALE
    keep = Rk > 1e-6
    return dict(R0=x[0] * SCALE, Rk=Rk[keep], tauk=B.tau[keep])


def sine_response(des, ecm, f, amp, idc, n_settle=None, ocv=355.0, i_noise=0.0, seed=0):
    """Run the cycle-averaged converter model with a sine current reference and return
    the complex gain i_bat/i_ref at f and harmonic amplitudes (relative) of i_bat."""
    Ts = des.Ts
    n_per = int(round(1 / (f * Ts)))
    f_true = 1 / (n_per * Ts)
    n_settle = n_settle or max(3, int(0.05 / (n_per * Ts)) + 1)
    n_meas = 4
    n = n_per * (n_settle + n_meas)
    k = np.arange(n)
    iref = idc + amp * np.sin(2 * np.pi * f_true * k * Ts)
    iL, ib, vb, d = ch.simulate_cycle_averaged(des, iref, ocv, ecm, i_noise=i_noise, seed=seed)
    sl = slice(n_per * n_settle, n)
    kk = k[sl]
    e = np.exp(-2j * np.pi * f_true * kk * Ts)
    Iref = 2 * np.mean((iref[sl] - idc) * e)
    Ib = 2 * np.mean(ib[sl] * e)
    harm = [abs(2 * np.mean(ib[sl] * np.exp(-2j * np.pi * h * f_true * kk * Ts))) / abs(Ib) for h in (2, 3, 4, 5)]
    return Ib / Iref, harm, (kk * Ts, iref[sl], ib[sl], vb[sl])


def main():
    out = {}
    i_ref_state = META.query("cell_id == 0 and severity == 0").index[0]
    Zc = D["Z"][i_ref_state, list(SOC).index(0.5)]
    des = ch.ConverterDesign()
    marg = ch.stability_margins(des, zfun(Zc))
    out["loop"] = dict(Kp=des.Kp, Ki=des.Ki, **marg, ripple_pp_A_at_355V=des.ripple_pp(355.0))
    print("Current loop:", out["loop"])
    ecm = pack_ecm(Zc)

    # ---------- A/B: analytic loop vs cycle-averaged time-domain model ----------
    fl = np.logspace(0, np.log10(4000), 9)
    rows = []
    for f in fl:
        g_sim, harm, _ = sine_response(des, ecm, f, amp=1.0, idc=7.5)
        g_an = ch.loop_transfer(des, np.array([f]), zfun(Zc)(np.array([f])))["i_bat_over_ref"][0]
        rows.append(dict(f_Hz=f, gain_sim=abs(g_sim), phase_sim_deg=np.degrees(np.angle(g_sim)),
                         gain_analytic=abs(g_an), phase_analytic_deg=np.degrees(np.angle(g_an))))
    LV = pd.DataFrame(rows).round(4)
    LV.to_csv(C.RESULTS / "phase2_loop_validation.csv", index=False)
    print(LV.to_string(index=False))

    # ---------- C: dead-time distortion vs DC current and amplitude (pack currents) ----------
    rows = []
    for idc in [0.0, 0.5, 1.5, 4.5]:
        for amp in [0.3, 1.5, 4.5]:
            g, harm, wav = sine_response(des, ecm, 10.0, amp=amp, idc=idc)
            thd = float(np.sqrt(np.sum(np.square(harm))))
            rows.append(dict(i_dc_pack_A=idc, amp_pk_pack_A=amp, crosses_zero=idc < amp + des.ripple_pp(355) / 2,
                             THD_ibat=thd, gain_error=abs(abs(g) - 1)))
    DT = pd.DataFrame(rows)
    DT.to_csv(C.RESULTS / "phase2_deadtime_distortion.csv", index=False)
    print(DT.round(5).to_string(index=False))
    out["deadtime"] = dict(max_THD_crossing=float(DT[DT.crosses_zero].THD_ibat.max()),
                           max_THD_no_crossing=float(DT[~DT.crosses_zero].THD_ibat.max()) if (~DT.crosses_zero).any() else None)

    # waveform figure (zero-mean injection at rest: crossing case)
    _, _, w0 = sine_response(des, ecm, 10.0, amp=1.5, idc=0.0)
    _, _, w1 = sine_response(des, ecm, 10.0, amp=1.5, idc=4.5)
    t_sw, iL_sw, vsw = ch._sim_switching(6, 400, des.Ts, des.V_dc, des.L, des.R_L, des.C_o, des.ESR_C,
                                         des.t_dead, 355.0 / 450.0, 355.0, ecm["R0"] + 0.5)
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.8))
    sl = slice(0, int(0.1 / des.Ts))
    axs[0].plot(w0[0][sl] - w0[0][0], w0[1][sl], color=INK2, lw=1, label="reference")
    axs[0].plot(w0[0][sl] - w0[0][0], w0[2][sl], color=CAT[1], lw=1.4, ls="--", label="battery current")
    axs[0].set_title("Zero-mean injection at rest (crosses 0 A)")
    axs[1].plot(w1[0][sl] - w1[0][0], w1[1][sl], color=INK2, lw=1, label="reference")
    axs[1].plot(w1[0][sl] - w1[0][0], w1[2][sl], color=CAT[0], lw=1.4, ls="--", label="battery current")
    axs[1].set_title("Injection on 4.5 A DC (no crossing)")
    for a in axs[:2]:
        a.set_xlabel("time (s)"); a.set_ylabel("pack current (A)"); a.legend(fontsize=6.5)
    n2 = 2 * 400
    axs[2].plot(t_sw[-n2:] * 1e6 - t_sw[-n2] * 1e6, iL_sw[-n2:], color=CAT[0], lw=1.2)
    axs[2].set_xlabel(r"time ($\mu$s)"); axs[2].set_ylabel("inductor current (A)")
    axs[2].set_title("Switching ripple, 2 periods (100 kHz)")
    fig.suptitle("Fig. 2b  Converter waveforms: dead-time distortion near zero current and PWM ripple", fontsize=8.5)
    fig.tight_layout(); save(fig, C.FIGURES / "fig2b_converter_waveforms.png")

    # ---------- E: CC-CV charge demo with EIS windows (pack ECM, 1 s steps) ----------
    ocv_cell = D["ocv"][i_ref_state]; Q = META.loc[i_ref_state, "Q"] * ch.N_PAR
    soc, t, I_cc, V_cv, dt = 0.10, 0.0, 0.5 * Q, 4.2 * ch.N_SERIES, 1.0
    R_dc = SCALE * float(np.real(Zc[np.argmin(np.abs(F - 1e-2))]))
    Kv, Kiv, integ = 0.3, 0.05, 0.0   # proportional gain x R_dc < 1, else the 1-s discrete loop limit-cycles
    log = []
    ocv_ext = lambda z: ch.N_SERIES * np.interp(z, np.r_[0.0, SOC, 1.0],
                                               np.r_[ocv_cell[0] - 0.25, ocv_cell, ocv_cell[-1] + 0.12])
    mode = "CC"
    while soc < 0.999 and t < 5 * 3600:
        v_oc = ocv_ext(soc)
        if mode == "CC":
            I = I_cc
            if v_oc + I * R_dc >= V_cv:
                mode = "CV"; integ = I_cc
        if mode == "CV":
            v = v_oc + I * R_dc
            e = V_cv - v
            integ += Kiv * e * dt
            I = float(np.clip(Kv * e + integ, 0, I_cc))
            if I < 0.05 * Q:
                break
        v = v_oc + I * R_dc
        soc += I * dt / 3600 / Q
        log.append((t, soc, I, v, mode))
        t += dt
    L = pd.DataFrame(log, columns=["t", "soc", "I", "V", "mode"])
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 2.8))
    ax[0].plot(L.t / 60, L.I, color=CAT[0]); ax[0].set_ylabel("pack current (A)")
    ax[1].plot(L.t / 60, L.V, color=CAT[1]); ax[1].set_ylabel("pack voltage (V)")
    t_cv = L[L["mode"] == "CV"].t.min() / 60
    for a in ax:
        a.axvline(t_cv, color=INK2, ls=":", lw=1); a.set_xlabel("time (min)")
    for s0 in C.SOC_TEST:
        tt = L.t[np.argmin(np.abs(L.soc - s0))] / 60
        ax[0].axvspan(tt, tt + ms.test_duration(ms.MeasConfig()) / 60, color=SEQ(0.35), alpha=0.6, lw=0)
    ax[0].text(t_cv + 1, 0.5 * I_cc, "CV", color=INK2); ax[0].text(2, 0.5 * I_cc, "CC (0.5 C)", color=INK2)
    fig.suptitle("Fig. 2c  CC-CV charge of the 96s3p pack (simplified ECM); shaded = EIS test windows at 30/50/70 % SOC (baseline test length)",
                 fontsize=8)
    fig.tight_layout(); save(fig, C.FIGURES / "fig2c_cccv_eis_windows.png")
    # current injected in CV mode is fought by the voltage loop: sensitivity 1/(1+G_v)
    fq = np.logspace(-4, 0, 200)
    s = 2j * np.pi * fq
    Gv = (Kv + Kiv / s) * R_dc   # voltage loop gain with a resistive battery approximation
    S = 1 / (1 + Gv)
    out["cccv"] = dict(t_charge_min=float(L.t.iloc[-1] / 60), t_cv_start_min=float(t_cv),
                       cv_sensitivity_at_10mHz=float(abs(np.interp(1e-2, fq, np.abs(S)))),
                       cv_sensitivity_at_1Hz=float(abs(S[-1])))
    print("CC-CV:", out["cccv"])

    # ---------- F: validate the fast measurement model against a sample-level simulation ----
    out["validation"] = validate_measurement_model(D["Z"][i_ref_state], D["ocv"][i_ref_state], META.loc[i_ref_state, "Q"])

    # ---------- G: measured vs clean spectra, baseline and stressed configurations -----------
    cfgs = {
        "baseline (rest, 0.5 A, 10 mHz-1 kHz, P=2)": ms.MeasConfig(),
        "low amplitude (0.05 A)": ms.MeasConfig(amp_pk_cell=0.05),
        "during C/3 charge, no detrend": ms.MeasConfig(i_dc_cell=5.0 / 3, detrend=False),
        "during C/3 charge, detrended": ms.MeasConfig(i_dc_cell=5.0 / 3),
        "50 us channel skew": ms.MeasConfig(skew=50e-6),
        "inductor sensing to 10 kHz": ms.MeasConfig(f_max=1e4),
    }
    fig, axs = plt.subplots(2, 3, figsize=(9.5, 5.6))
    rows = []
    rng = np.random.default_rng(C.SEED + 21)
    for ax, (name, cfg) in zip(axs.flat, cfgs.items()):
        m = ms.measure(D["Z"][i_ref_state], F, SOC, D["ocv"][i_ref_state], META.loc[i_ref_state, "Q"], 0.5, cfg, rng)
        ztrue = ms._interp_logf(F, D["Z"][i_ref_state][[list(SOC).index(0.5)]], m["f"])[0]
        err = (m["Z"] - ztrue) / np.abs(ztrue)
        rows.append(dict(config=name, duration_min=m["duration_s"] / 60, dSOC=m["dsoc"], n_lines=len(m["f"]),
                         median_rel_err=float(np.median(np.abs(err))), max_rel_err=float(np.max(np.abs(err))),
                         median_sigma_rel=float(np.median(m["sigma"] / np.abs(ztrue)))))
        zz = D["Z"][i_ref_state, list(SOC).index(0.5)]
        ax.plot(zz.real * 1e3, -zz.imag * 1e3, color=INK2, lw=1.2, label="clean (lab)")
        ax.errorbar(m["Z"].real * 1e3, -m["Z"].imag * 1e3, xerr=m["sigma"] * 1e3, yerr=m["sigma"] * 1e3,
                    fmt="o", ms=3, color=CAT[1], ecolor=CAT[1], elinewidth=0.6, label="charger estimate")
        ax.set_title(name, fontsize=7.5)
        ax.set_xlabel(r"Z' (m$\Omega$)"); ax.set_ylabel(r"-Z'' (m$\Omega$)")
        ax.set_xlim(0, 75); ax.set_ylim(-5, 30)
    axs[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Fig. 2d  Charger-measured vs clean spectra (cell 0, BOL, 50 % SOC); error bars = per-line std estimated by the charger",
                 fontsize=8)
    fig.tight_layout(); save(fig, C.FIGURES / "fig2d_measured_vs_clean.png")
    MV = pd.DataFrame(rows).round(5)
    MV.to_csv(C.RESULTS / "phase2_measured_vs_clean.csv", index=False)
    print(MV.to_string(index=False))

    # loop Bode figure
    fq = np.logspace(0, 4.5, 400)
    lt = ch.loop_transfer(des, fq, zfun(Zc)(fq))
    lt_b = ch.loop_transfer(replace(des, sense_location="battery"), fq, zfun(Zc)(fq))
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 2.9))
    axs[0].semilogx(fq, 20 * np.log10(np.abs(lt["T_loop"])), color=CAT[0], label="loop gain |T|")
    axs[0].semilogx(fq, 20 * np.log10(np.abs(lt["i_bat_over_ref"])), color=CAT[1], ls="--", label=r"|i$_{bat}$/i$_{ref}$|")
    axs[0].axhline(0, color=INK2, lw=0.8); axs[0].set_ylabel("dB"); axs[0].set_xlabel("frequency (Hz)")
    axs[0].plot(LV.f_Hz, 20 * np.log10(LV.gain_sim), "o", color=CAT[1], ms=4, label="time-domain check")
    axs[0].legend(); axs[0].set_ylim(-30, 60)
    axs[1].semilogx(fq, np.degrees(np.angle(lt["bat_over_sensed"])), color=CAT[2], label="inductor sensing")
    axs[1].semilogx(fq, np.degrees(np.angle(lt_b["bat_over_sensed"])), color=CAT[0], ls="--", label="battery-side sensing")
    axs[1].set_ylabel(r"phase of i$_{bat}$/i$_{sensed}$ (deg)"); axs[1].set_xlabel("frequency (Hz)"); axs[1].legend()
    fig.suptitle(f"Fig. 2a  Current loop: crossover {marg['f_cross']/1e3:.2f} kHz, PM {marg['phase_margin_deg']:.0f} deg, "
                 f"GM {marg['gain_margin_dB']:.1f} dB; sensing-location error grows above ~1 kHz", fontsize=8)
    fig.tight_layout(); save(fig, C.FIGURES / "fig2a_current_loop.png")
    json.dump(out, open(C.RESULTS / "phase2_summary.json", "w"), indent=2, default=float)


def validate_measurement_model(Zcell_soc, ocv, Q, n_rep=40):
    """Sample-level time-domain simulation of one band set (0.1 Hz - 1 kHz) vs the fast model.
    Time-domain: periodic multisine current through the closed-loop transfer, battery voltage
    synthesised in the frequency domain (exact for periodic steady state, linear cell),
    white noise + real ADC rounding on both channels, AA filters with tolerance, fractional
    skew, OCV ramp from DC current, linear detrend, DFT per period, averaging."""
    cfg = ms.MeasConfig(f_min=0.1, f_max=1000.0, i_dc_cell=0.5, n_periods=2, f_flicker=0.0,
                         include_nonlinear=False)
    des = cfg.converter
    rng = np.random.default_rng(C.SEED + 7)
    unit = dict(dv=0.03, di=-0.03)
    soc0 = 0.5
    Zl_all = {}
    td_err, fm_err = [], []
    a_line = cfg.amp_pk_cell / ms.unit_peak()
    lsb_v = cfg.v_range / 2 ** cfg.v_bits; lsb_i = cfg.i_range_pack / 2 ** cfg.i_bits
    rate = cfg.i_dc_cell / (3600 * Q)
    for rep in range(n_rep):
        t_start = 0.0
        res_td, res_fm = [], []
        mfm = ms.measure(Zcell_soc, F, SOC, ocv, Q, soc0, cfg, rng, unit=unit)
        for f_lo, T0, lines in ms.band_plan(cfg):
            fs = cfg.fs
            N0 = int(round(T0 * fs)); P = cfg.n_periods; N = N0 * (P + 1)
            t = t_start + np.arange(N) / fs
            soc_t = soc0 + rate * t
            Zl = ms._interp_logf(F, Zcell_soc, lines)
            Zmid = np.array([np.interp(soc_t.mean(), SOC, Zl[:, j].real) + 1j * np.interp(soc_t.mean(), SOC, Zl[:, j].imag)
                             for j in range(len(lines))])
            lt = ch.loop_transfer(des, lines, Zmid * ch.N_SERIES / ch.N_PAR)
            Ib = a_line * lt["i_bat_over_ref"]
            Is = Ib / lt["bat_over_sensed"]
            n_l = len(lines)
            ph = -np.pi * np.arange(1, n_l + 1) * np.arange(n_l) / n_l
            Hv = ms._butter2(lines, cfg.aa_fc * (1 + unit["dv"])) / ms._butter2(lines, cfg.aa_fc)
            Hi = ms._butter2(lines, cfg.aa_fc * (1 + unit["di"])) / ms._butter2(lines, cfg.aa_fc)
            arg = 2 * np.pi * lines[:, None] * (t[None, :] - t_start) + ph[:, None]
            i_meas = (np.abs(Hi * Is)[:, None] * np.cos(arg + np.angle(Hi * Is)[:, None]
                                                         - 2 * np.pi * lines[:, None] * cfg.skew)).sum(0)
            v_ac = (np.abs(Hv * Zmid * Ib)[:, None] * np.cos(arg + np.angle(Hv * Zmid * Ib)[:, None])).sum(0)
            v = np.interp(soc_t, SOC, ocv) + v_ac
            scale_fs = np.sqrt(fs / cfg.fs)   # keep the noise PSD equal to the 25 kHz case
            vn = v + rng.normal(0, cfg.sigma_v * scale_fs, N)
            in_ = (i_meas * ch.N_PAR + rng.normal(0, np.sqrt(cfg.sigma_i_pack ** 2 + cfg.ripple_alias_pack ** 2) * scale_fs, N))
            vq = np.round(vn / lsb_v) * lsb_v
            iq = np.round(in_ / lsb_i) * lsb_i / ch.N_PAR
            sl = slice(N0, N)
            tt, vv, ii = t[sl], vq[sl], iq[sl]
            # joint least squares per period: offset + linear trend + cos/sin at every line
            # (a separate linear detrend would remove part of the multisine and bias low lines)
            Zp = []
            for p in range(P):
                s2 = slice(p * N0, (p + 1) * N0)
                tp = tt[s2] - t_start
                Bm = np.hstack([np.ones((N0, 1)), (tp - tp.mean())[:, None],
                                np.cos(2 * np.pi * lines[None, :] * tp[:, None]),
                                np.sin(2 * np.pi * lines[None, :] * tp[:, None])])
                cv = np.linalg.lstsq(Bm, vv[s2], rcond=None)[0]
                ci = np.linalg.lstsq(Bm, ii[s2], rcond=None)[0]
                Vk = cv[2:2 + n_l] - 1j * cv[2 + n_l:]
                Ik = ci[2:2 + n_l] - 1j * ci[2 + n_l:]
                Zp.append(Vk / Ik)
            Zhat = np.mean(Zp, 0)
            res_td.append((lines, Zhat, Zmid))
            t_start += (P + 1) * T0
        f_td = np.concatenate([r[0] for r in res_td]); o = np.argsort(f_td)
        Z_td = np.concatenate([r[1] for r in res_td])[o]
        Z_ref = np.concatenate([r[2] for r in res_td])[o]
        td_err.append((Z_td - Z_ref) / np.abs(Z_ref))
        fm_err.append((mfm["Z"] - mfm["Z_true_avg"]) / np.abs(mfm["Z_true_avg"]))
    td_err, fm_err = np.array(td_err), np.array(fm_err)
    f_l = np.sort(np.concatenate([b[2] for b in ms.band_plan(cfg)]))
    V = pd.DataFrame(dict(f_Hz=f_l,
                          bias_re_td=td_err.real.mean(0), bias_re_fast=fm_err.real.mean(0),
                          bias_im_td=td_err.imag.mean(0), bias_im_fast=fm_err.imag.mean(0),
                          std_td=np.sqrt((td_err.real.var(0) + td_err.imag.var(0)) / 2),
                          std_fast=np.sqrt((fm_err.real.var(0) + fm_err.imag.var(0)) / 2)))
    V.to_csv(C.RESULTS / "phase2_fast_model_validation.csv", index=False)
    ratio = V.std_fast / V.std_td
    summ = dict(n_rep=n_rep, std_ratio_median=float(ratio.median()), std_ratio_min=float(ratio.min()),
                std_ratio_max=float(ratio.max()),
                max_abs_bias_diff=float(np.max(np.abs(np.r_[V.bias_re_td - V.bias_re_fast, V.bias_im_td - V.bias_im_fast]))))
    print("Fast-model validation:", summ)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 2.8))
    axs[0].loglog(V.f_Hz, V.std_td * 100, "o", color=CAT[0], ms=4, label="sample-level simulation")
    axs[0].loglog(V.f_Hz, V.std_fast * 100, "-", color=CAT[1], label="fast model")
    axs[0].set_ylabel("random error, 1 std (% of |Z|)"); axs[0].set_xlabel("frequency (Hz)"); axs[0].legend()
    axs[1].semilogx(V.f_Hz, V.bias_im_td * 100, "o", color=CAT[0], ms=4, label="sample-level, Im")
    axs[1].semilogx(V.f_Hz, V.bias_im_fast * 100, "-", color=CAT[1], label="fast model, Im")
    axs[1].semilogx(V.f_Hz, V.bias_re_td * 100, "s", color=CAT[2], ms=4, label="sample-level, Re")
    axs[1].semilogx(V.f_Hz, V.bias_re_fast * 100, ":", color=CAT[3], label="fast model, Re")
    axs[1].set_ylabel("systematic error (% of |Z|)"); axs[1].set_xlabel("frequency (Hz)"); axs[1].legend(fontsize=6)
    fig.suptitle(f"Fig. 2e  Fast measurement model vs sample-level time-domain simulation ({n_rep} repeats, "
                 "0.1 Hz-1 kHz, 0.5 A, C/10 DC, 3 % filter mismatch, 5 us skew)", fontsize=8)
    fig.tight_layout(); save(fig, C.FIGURES / "fig2e_fast_model_validation.png")
    return summ


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f} s")
