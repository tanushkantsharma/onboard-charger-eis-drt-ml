"""Distribution of relaxation times (DRT): Tikhonov (NNLS) and Bayesian (Laplace / Gibbs).

Model:  Z(w) = R_inf + j w L + 1/(j w C_s) + sum_m g_m * phi_m(w)
        phi_m(w) = integral  rbf_m(y) / (1 + j w e^y) dy,   y = ln(tau)
Unknowns x = [R_inf, L, 1/C_s, g_1..g_M] >= 0.
  * R_inf  : ohmic resistance (electrolyte, current collectors, tabs)
  * L      : series inductance (cell + cables)
  * 1/C_s  : series capacitance -> absorbs the purely capacitive (dOCV/dQ) low-frequency
             branch so that it does not pile up at the right edge of the DRT
  * g_m    : weights of Gaussian radial basis functions on a log-tau grid (DRTtools-style,
             Wan et al. 2015). gamma(y) = sum g_m rbf_m(y)  [Ohm per unit ln(tau)]
Tikhonov: minimise ||W (A x - Z)||^2 + lam ||D g||^2 subject to x >= 0  (solved as NNLS)
Bayesian: the same objective is the negative log posterior of a Gaussian likelihood with
          per-frequency std sigma_k (W = 1/sigma) and a Gaussian smoothness prior; positivity
          makes the posterior a truncated Gaussian. We sample it (a) with a fast Laplace
          approximation around the MAP on the active set, and (b) with an exact Gibbs sampler
          used only to validate (a).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls
from scipy.stats import truncnorm


class DRTBasis:
    def __init__(self, freqs, tau_min=None, tau_max=None, pts_per_dec=10, fwhm_coeff=0.5,
                 n_quad=401):
        freqs = np.asarray(freqs, float)
        self.freqs = freqs
        w = 2 * np.pi * freqs
        if tau_min is None:
            tau_min = 1.0 / (2 * np.pi * freqs.max())   # no extrapolation above f_max: keeps R_inf identifiable
        if tau_max is None:
            tau_max = 1.0 / (2 * np.pi * freqs.min()) * 10
        n = int(np.round(np.log10(tau_max / tau_min) * pts_per_dec)) + 1
        self.y = np.linspace(np.log(tau_min), np.log(tau_max), n)   # ln(tau) centres
        self.tau = np.exp(self.y)
        dy = self.y[1] - self.y[0]
        self.dy = dy
        # Gaussian RBF: exp(-(eps*(y-ym))^2), FWHM = fwhm_coeff * spacing^-1 ... keep simple:
        fwhm = dy / fwhm_coeff
        self.eps = 2 * np.sqrt(np.log(2)) / fwhm
        # numerical integration of each basis function's impedance
        u = np.linspace(-6 / self.eps, 6 / self.eps, n_quad)
        du = u[1] - u[0]
        rbf = np.exp(-(self.eps * u) ** 2)
        Phi = np.empty((len(w), n), complex)
        for m, ym in enumerate(self.y):
            t = np.exp(ym + u)
            Phi[:, m] = (rbf[None, :] / (1 + 1j * w[:, None] * t[None, :])).sum(1) * du
        self.Phi = Phi
        self.area = np.sqrt(np.pi) / self.eps            # integral of one RBF
        A = np.zeros((len(w), n + 3), complex)
        A[:, 0] = 1.0
        A[:, 1] = 1j * w
        A[:, 2] = 1.0 / (1j * w)
        A[:, 3:] = Phi
        self.A = A
        # second-difference regulariser on the RBF weights only
        D = np.zeros((n - 2, n + 3))
        for i in range(n - 2):
            D[i, 3 + i:3 + i + 3] = [1, -2, 1]
        D1 = np.zeros((n - 1, n + 3))
        for i in range(n - 1):
            D1[i, 3 + i:3 + i + 2] = [-1, 1]
        self.D = np.vstack([D, 0.1 * D1])
        self.n = n
        # column scaling so that NNLS is well conditioned
        self.colscale = np.ones(n + 3)
        self.colscale[1] = 1e-7      # L in H  (~1e-8..1e-7)
        self.colscale[2] = 1e-1      # 1/C in 1/F (C ~ 1e3..1e5 F -> 1/C ~ 1e-5..1e-3)

    def gamma(self, g, y_eval=None):
        """Evaluate gamma(ln tau) [Ohm] on y_eval (default: basis centres)."""
        y_eval = self.y if y_eval is None else y_eval
        return (np.exp(-(self.eps * (y_eval[:, None] - self.y[None, :])) ** 2) * g[None, :]).sum(-1)

    def _real_system(self, Z, sigma):
        W = 1.0 / np.asarray(sigma)
        Ar = np.vstack([self.A.real * W[:, None], self.A.imag * W[:, None]]) * self.colscale[None, :]
        b = np.concatenate([Z.real * W, Z.imag * W])
        return Ar, b

    def fit(self, Z, sigma=None, lam=1e4, mask=None):
        """Tikhonov-regularised non-negative fit. sigma: per-frequency std of Re and Im (Ohm).
        If sigma is None, relative weighting sigma = |Z| is used. mask: boolean frequency subset."""
        Z = np.asarray(Z)
        sigma = np.abs(Z) if sigma is None else np.broadcast_to(np.asarray(sigma, float), Z.shape)
        m = np.ones(len(Z), bool) if mask is None else np.asarray(mask, bool)
        W = 1.0 / sigma[m]
        A = self.A[m]
        Ar = np.vstack([A.real * W[:, None], A.imag * W[:, None]]) * self.colscale[None, :]
        b = np.concatenate([Z[m].real * W, Z[m].imag * W])
        # Absolute (Bayesian-consistent) regulariser: lam = 1/prior_scale^2, prior_scale in Ohm.
        # With W = 1/sigma this is exactly -2 log posterior of a Gaussian likelihood and a
        # Gaussian smoothness prior; more/better data therefore automatically means less
        # relative smoothing and narrower posteriors.
        Dd = self.D * self.colscale[None, :]
        # weak Gaussian prior on L (std 100 nH in scaled units) so that L stays bounded when
        # the band stops well below the inductive region
        Lrow = np.zeros((1, Ar.shape[1])); Lrow[0, 1] = 1.0
        K = np.vstack([Ar, np.sqrt(lam) * Dd, Lrow])
        rhs = np.concatenate([b, np.zeros(Dd.shape[0] + 1)])
        xs, _ = nnls(K, rhs, maxiter=5000)
        x = xs * self.colscale
        self._last = dict(K=K, rhs=rhs, xs=xs, nrow=len(b))
        return x

    def predict(self, x, freqs_mask=None):
        A = self.A if freqs_mask is None else self.A[freqs_mask]
        return A @ x

    # ---------------- Bayesian: Laplace approximation ----------------
    def posterior_samples_laplace(self, n_samples=50, rng=None, tol=1e-12):
        """Gaussian posterior around the MAP restricted to the active (positive) set,
        samples clipped at zero. Requires a preceding call to fit() with real sigma."""
        rng = np.random.default_rng(rng)
        K, xs = self._last["K"], self._last["xs"]
        active = xs > tol * max(xs.max(), 1e-30)
        H = K[:, active].T @ K[:, active]        # precision (sigma already in K)
        try:
            Lc = np.linalg.cholesky(H + 1e-12 * np.eye(H.shape[0]) * np.trace(H) / H.shape[0])
        except np.linalg.LinAlgError:
            Lc = np.linalg.cholesky(H + 1e-8 * np.eye(H.shape[0]) * np.trace(H) / H.shape[0])
        z = rng.standard_normal((active.sum(), n_samples))
        dx = np.linalg.solve(Lc.T, z)             # cov = H^-1
        S = np.repeat(xs[:, None], n_samples, 1)
        S[active] += dx
        S = np.clip(S, 0, None)
        return (S * self.colscale[:, None]).T     # [n_samples, n_params]

    # ---------------- Bayesian: exact Gibbs, compiled (main sampler) ----------------
    def posterior_samples(self, n_samples=30, burn=300, thin=10, rng=None):
        """Exact Gibbs sampler of the truncated-Gaussian posterior (all x >= 0), numba-compiled.
        Each coordinate is drawn from its 1-D truncated-normal full conditional."""
        rng = np.random.default_rng(rng)
        K, rhs, xs = self._last["K"], self._last["rhs"], self._last["xs"]
        H = K.T @ K
        h = K.T @ rhs
        S = _gibbs_tn(H, h, xs.copy(), n_samples, burn, thin, int(rng.integers(1 << 31)))
        return S * self.colscale[None, :]

    # ---------------- Bayesian: exact Gibbs (scipy reference, slow) ----------------
    def posterior_samples_gibbs(self, n_samples=200, burn=200, thin=2, rng=None):
        """Exact Gibbs sampler of the truncated-Gaussian posterior (x >= 0)."""
        rng = np.random.default_rng(rng)
        K, rhs, xs = self._last["K"], self._last["rhs"], self._last["xs"].copy()
        H = K.T @ K
        h = K.T @ rhs
        d = np.diag(H).copy()
        x = xs.copy()
        out = []
        for it in range(burn + n_samples * thin):
            for i in range(len(x)):
                mu = (h[i] - H[i] @ x + d[i] * x[i]) / d[i]
                sd = 1 / np.sqrt(d[i])
                a = (0 - mu) / sd
                x[i] = truncnorm.rvs(a, np.inf, loc=mu, scale=sd, random_state=rng)
            if it >= burn and (it - burn) % thin == 0:
                out.append(x.copy())
        return np.array(out) * self.colscale[None, :]


def drt_features(basis: DRTBasis, x, edges_log10tau=None):
    """Compact, physically interpretable features from one DRT solution.
    - R_inf, L, log10(1/C_s)
    - resistance contained in each log10(tau) decade band (area under gamma)
    """
    if edges_log10tau is None:
        edges_log10tau = np.arange(-5, 4.01, 1.0)        # 1e-5 s .. 1e4 s, decade bands
    g = x[3:]
    yf = np.linspace(basis.y[0], basis.y[-1], 600)
    gam = basis.gamma(g, yf)
    l10 = yf / np.log(10)
    feats = [x[0], x[1] * 1e9, np.log10(x[2] + 1e-9)]
    dyf = yf[1] - yf[0]
    for lo, hi in zip(edges_log10tau[:-1], edges_log10tau[1:]):
        sel = (l10 >= lo) & (l10 < hi)
        feats.append(gam[sel].sum() * dyf)
    return np.array(feats)


def feature_names(edges_log10tau=None):
    if edges_log10tau is None:
        edges_log10tau = np.arange(-5, 4.01, 1.0)
    names = ["R_inf", "L_nH", "log10_invC"]
    names += [f"R_tau[1e{int(a)},1e{int(b)})" for a, b in zip(edges_log10tau[:-1], edges_log10tau[1:])]
    return names


from numba import njit
import math


@njit(cache=True)
def _rtn_lower(mu, sd):
    """Draw from N(mu, sd^2) truncated to [0, inf) (Robert 1995 exponential rejection)."""
    a = -mu / sd
    if a < 0.45:
        while True:
            z = np.random.randn()
            if z >= a:
                return mu + sd * z
    lam = 0.5 * (a + math.sqrt(a * a + 4.0))
    while True:
        z = a - math.log(np.random.random()) / lam
        if np.random.random() <= math.exp(-0.5 * (z - lam) ** 2):
            return mu + sd * z


@njit(cache=True)
def _gibbs_tn(H, h, x, n_samples, burn, thin, seed):
    np.random.seed(seed)
    n = x.shape[0]
    out = np.empty((n_samples, n))
    Hx = H @ x
    k = 0
    for it in range(burn + n_samples * thin):
        for i in range(n):
            d = H[i, i]
            mu = (h[i] - Hx[i] + d * x[i]) / d
            xn = _rtn_lower(mu, 1.0 / math.sqrt(d))
            dx = xn - x[i]
            if dx != 0.0:
                for j in range(n):
                    Hx[j] += H[j, i] * dx
                x[i] = xn
        if it >= burn and (it - burn) % thin == 0:
            out[k] = x
            k += 1
    return out
