"""Phase 1: physics-based ground truth.

Each *virtual cell* = Chen2020 (LG M50) parameters + random manufacturing spread.
Each cell is assigned ONE dominant degradation mechanism and aged through a fixed
list of severities. Ageing is emulated at the parameter level ("degradation-mode"
approach, cf. Birkl et al. 2017; O'Kane et al. 2022) - we do NOT run cycle-ageing
simulations. This is a modelling choice that must be stated in the paper.

Mechanism -> parameter changes (s = severity in [0, 1]):
  SEI     : SEI film thickness grows (film resistance in series with the negative
            interface) + loss of lithium inventory (LLI) + small porosity loss.
  PLATING : LLI (plated/dead Li) + strong porosity loss in the negative electrode
            concentrated near the separator (pore clogging) + mild extra SEI.
  LAM     : loss of active material in negative and/or positive electrode
            (smaller active surface and capacity). Lithium inside lost negative
            particles is lost too (handled automatically by PyBaMM's cyclable-Li count).
"""
from __future__ import annotations
import warnings
import numpy as np
import pybamm
from . import config as C

warnings.filterwarnings("ignore")
pybamm.set_logging_level("ERROR")


def _scale(pv, key, factor):
    """Multiply a parameter that may be a constant or a function by `factor`."""
    v = pv[key]
    if callable(v):
        pv.update({key: (lambda *a, _v=v, _f=factor: _f * _v(*a))})
    else:
        pv.update({key: v * factor})


def sample_cell(rng: np.random.Generator, mechanism: str, cell_id: int) -> dict:
    """Draw manufacturing spread + mechanism-specific ageing rates for one cell."""
    ln = lambda s: float(np.exp(rng.normal(0.0, s)))
    cell = dict(cell_id=cell_id, mechanism=mechanism,
                # manufacturing spread (log-normal multipliers)
                m_Rn=ln(0.08), m_Rp=ln(0.08), m_Dn=ln(0.20), m_Dp=ln(0.20),
                m_j0n=ln(0.15), m_j0p=ln(0.15), m_Cdln=ln(0.15), m_Cdlp=ln(0.15),
                m_Lsei0=ln(0.15), m_kappa=ln(0.08),
                R_ext=float(rng.uniform(1.0e-3, 3.0e-3)),   # tabs/welds/contacts [Ohm]
                L_ext=float(rng.uniform(15e-9, 40e-9)),     # cell + lead inductance [H]
                sev_jitter=float(rng.uniform(0.85, 1.15)))
    if mechanism == "SEI":
        cell.update(sei_growth=rng.uniform(10, 25), lli_max=rng.uniform(0.10, 0.20),
                    eps_loss=rng.uniform(0.0, 0.04))
    elif mechanism == "PLATING":
        cell.update(lli_max=rng.uniform(0.10, 0.22), pore_loss=rng.uniform(0.35, 0.60),
                    pore_len=rng.uniform(0.10, 0.25), sei_growth=rng.uniform(1, 4))
    elif mechanism == "LAM":
        cell.update(lam_max=rng.uniform(0.15, 0.30), lam_share_n=rng.uniform(0.0, 1.0))
    else:
        raise ValueError(mechanism)
    return cell


def build_parameters(cell: dict, severity: float) -> pybamm.ParameterValues:
    pv = pybamm.ParameterValues(C.PARAM_SET)
    # ---------- manufacturing spread ----------
    _scale(pv, "Negative particle radius [m]", cell["m_Rn"])
    _scale(pv, "Positive particle radius [m]", cell["m_Rp"])
    _scale(pv, "Negative particle diffusivity [m2.s-1]", cell["m_Dn"])
    _scale(pv, "Positive particle diffusivity [m2.s-1]", cell["m_Dp"])
    _scale(pv, "Negative electrode exchange-current density [A.m-2]", cell["m_j0n"])
    _scale(pv, "Positive electrode exchange-current density [A.m-2]", cell["m_j0p"])
    _scale(pv, "Negative electrode double-layer capacity [F.m-2]", cell["m_Cdln"])
    _scale(pv, "Positive electrode double-layer capacity [F.m-2]", cell["m_Cdlp"])
    _scale(pv, "Initial SEI thickness [m]", cell["m_Lsei0"])
    _scale(pv, "Electrolyte conductivity [S.m-1]", cell["m_kappa"])

    s = float(np.clip(severity * cell["sev_jitter"], 0.0, 1.0))
    mech = cell["mechanism"]
    if mech == "SEI":
        _scale(pv, "Initial SEI thickness [m]", 1.0 + (cell["sei_growth"] - 1.0) * s)
        _scale(pv, "Initial concentration in negative electrode [mol.m-3]", 1.0 - cell["lli_max"] * s)
        _scale(pv, "Negative electrode porosity", 1.0 - cell["eps_loss"] * s)
    elif mech == "PLATING":
        _scale(pv, "Initial concentration in negative electrode [mol.m-3]", 1.0 - cell["lli_max"] * s)
        _scale(pv, "Initial SEI thickness [m]", 1.0 + (cell["sei_growth"] - 1.0) * s)
        e0 = pv["Negative electrode porosity"]
        Ln = pv["Negative electrode thickness [m]"]
        a, lam = cell["pore_loss"] * s, cell["pore_len"] * Ln

        def eps_n(x, *args, _e0=e0, _a=a, _lam=lam, _L=Ln):
            return _e0 * (1.0 - _a * pybamm.exp(-(_L - x) / _lam))
        pv.update({"Negative electrode porosity": eps_n})
    elif mech == "LAM":
        tot = cell["lam_max"] * s
        _scale(pv, "Negative electrode active material volume fraction", 1.0 - tot * cell["lam_share_n"])
        _scale(pv, "Positive electrode active material volume fraction", 1.0 - tot * (1 - cell["lam_share_n"]))
    return pv


_PARAM = pybamm.LithiumIonParameters()


def capacity_and_ocv(pv, soc_grid):
    """Usable capacity [Ah] from the electrode-SOH solver and OCV(SOC) [V]."""
    esoh = pybamm.lithium_ion.ElectrodeSOHSolver(pv, _PARAM)
    inp = {"V_min": pv["Lower voltage cut-off [V]"], "V_max": pv["Upper voltage cut-off [V]"],
           "Q_n": pv.evaluate(_PARAM.n.Q_init), "Q_p": pv.evaluate(_PARAM.p.Q_init),
           "Q_Li": pv.evaluate(_PARAM.Q_Li_particles_init)}
    sol = esoh.solve(inp)
    x0, x100, y0, y100 = (float(sol[k]) for k in ("x_0", "x_100", "y_0", "y_100"))
    T = pv["Reference temperature [K]"]
    Un, Up = pv["Negative electrode OCP [V]"], pv["Positive electrode OCP [V]"]
    ocv = []
    for z in soc_grid:
        x = x0 + z * (x100 - x0)
        y = y0 - z * (y0 - y100)
        ocv.append(float(pv.evaluate(Up(pybamm.Scalar(y)))) - float(pv.evaluate(Un(pybamm.Scalar(x)))))
    return float(sol["Q"]), np.array(ocv), dict(Q_n=float(inp["Q_n"]), Q_p=float(inp["Q_p"]),
                                                Q_Li=float(inp["Q_Li"]))


def simulate_state(cell: dict, severity: float, freqs=C.FREQS, soc_grid=C.SOC_GRID):
    """Clean impedance Z[soc, f] (complex, Ohm) incl. external R and L, plus labels."""
    pv = build_parameters(cell, severity)
    Q, ocv, qs = capacity_and_ocv(pv, soc_grid)
    model = pybamm.lithium_ion.DFN(options=C.PYBAMM_OPTIONS)
    sim = pybamm.EISSimulation(model, parameter_values=pv)
    Z = np.empty((len(soc_grid), len(freqs)), dtype=complex)
    w = 2 * np.pi * freqs
    for i, z in enumerate(soc_grid):
        Z[i] = sim.solve(freqs, initial_soc=float(z)).impedance + cell["R_ext"] + 1j * w * cell["L_ext"]
    return Z, Q, ocv, qs
