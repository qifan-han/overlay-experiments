# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "scipy>=1.12",
#   "matplotlib>=3.8",
#   "pandas>=2.2",
# ]
# ///
"""
Monte Carlo Simulations: "Measuring Causal Effects under Opaque Targeting"

Three DGPs vary the platform's *opaque* targeting rule from simple to complex,
following Ye et al. (2025) who vary the outcome model complexity across DGPs.
Here the analogue is complexity of the assignment rule — the part the analyst
never observes.  In all three DGPs the MTE schedule is monotone decreasing so
Assumptions 4 (monotone selection) and 5 (nonneg) both hold; the key quantity
that changes is e(x) = P(D=1|X=x) and how it relates to treatment effects.

For each DGP we evaluate:
  Track B (Section 3.5.1): plug-in bounds [CITT,CATT] and Imbens-Manski /
  Stoye CI covering the true CATE.  Metrics: CATT bias/RMSE, CI coverage and
  width, Selection Ratio upper bound.

  Track A (Section 3.5.2): finite-differences MTE estimator applied to DGP 1
  extended with a discrete geographic excluded shifter.  Metrics: MTE bias at
  each propensity midpoint, CATE RMSE.

Two robustness DGPs show what breaks when assumptions are violated:
  DGP-R1: MTE hump-shaped (monotone selection violated) → upper bound may fail.
  DGP-R2: MTE crosses zero (nonneg violated) → lower bound fails.

─────────────────────────────────────────────────────────────────────────────
DGP Details
─────────────────────────────────────────────────────────────────────────────
DGP 1  Constant-quota targeting  (simplest opaque rule)
       D_i = 1{U_i ≤ 0.30}  (platform targets bottom 30% by latent type)
       τ(U_i) = 2(1 − U_i)   (monotone: low-U → high-responders, targeted)
       True (analytic):
         CATE = 1.00,  CATT = 1.76,  CITT = 0.528,  SR = 1.76,  SR_ub = 3.33

DGP 2  Logistic score-based targeting  (realistic one-covariate rule)
       D_i = 1{U_i ≤ σ(α + βX_i)},  X_i ~ N(0,1),  α calibrated to ē = 0.30
       τ(U_i) = 2(1 − U_i)   (same monotone MTE as DGP 1)
       High-X users have higher propensity AND are more likely to be low-U
       compliers → CATT > CATE, same qualitative picture as DGP 1.
       True CATT > CATE by simulation.

DGP 3  Nonlinear ML targeting  (complex two-covariate interaction rule)
       D_i = 1{U_i ≤ σ(γ₀+γ₁X₁²+γ₂X₂+γ₃X₁X₂)},  X₁,X₂~N(0,1),  ē≈0.30
       τ(U_i) = 2(1 − U_i)   (same monotone MTE)
       Tests robustness of the estimator to arbitrary nonlinear opaque rules.

DGP-R1 Hump-shaped MTE, constant-quota targeting
       τ(U_i) = 8U_i(1 − U_i)  — peaks at U=0.5, monotone selection violated.
       → Upper bound CATE ≤ CATT may fail (CATE > CATT here).

DGP-R2 Linear-crossing MTE, constant-quota targeting
       τ(U_i) = 2 − 4U_i  — positive for U<0.5, negative for U>0.5.
       → Nonneg assumption violated; lower bound CATE ≥ CITT fails.

Track A extension of DGP 1:  geographic markets S_i ∈ {1,2,3} with
  constant propensities π_s ∈ {0.20, 0.35, 0.55}.  In market s,
  D_i = 1{U_i ≤ π_s}.  True MTE(p) = 2(1−p), CATE = 1.00.
─────────────────────────────────────────────────────────────────────────────
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import brentq
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ─────────────────────────────────────────────────────────────────────────────
# Section 3.5.1 Track B: plug-in bounds + Imbens-Manski (2004) / Stoye (2009) CI
# ─────────────────────────────────────────────────────────────────────────────

def track_b(Y, Z, W, alpha=0.10):
    """
    Plug-in estimators for CITT, CATT, and the Imbens-Manski CI.

    CATT̂  = (Ȳ_{Z=1} − Ȳ_{Z=0}) / P̂(W=1|Z=1)

    Variance of CATT̂ via delta method (eq. delta_var in paper):
      σ²_U ≈ σ²_CITT/ê² + CATT²·σ²_e/ê² − 2·CATT·Ĉov(L̂,ê)/ê³
    where Ĉov(L̂,ê) = sample_Cov(Y,W|Z=1)/n₁.

    Imbens-Manski CI: solve Φ(ĉ + √n·(Û−L̂)/max(σ̂_L,σ̂_U)) − Φ(−ĉ) = 1−α.
    CI = [L̂ − ĉ·σ̂_L/√n,  Û + ĉ·σ̂_U/√n].
    """
    m1 = Z == 1;  m0 = Z == 0
    n1, n0 = m1.sum(), m0.sum()
    n  = n1 + n0
    Y1, Y0, W1 = Y[m1], Y[m0], W[m1]

    # ── Bound estimators ────────────────────────────────────────────────────
    CITT  = Y1.mean() - Y0.mean()          # lower bound L̂
    e_hat = W1.mean()
    if e_hat < 1e-9:
        nan = float('nan')
        return dict(CITT=CITT, CATT=nan, e_hat=e_hat, sigma_L=nan, sigma_U=nan,
                    ci_lo=nan, ci_hi=nan, width=nan, SR_ub=nan)
    CATT = CITT / e_hat                    # upper bound Û

    # ── Delta-method variance ────────────────────────────────────────────────
    sigma_CITT_sq = Y1.var(ddof=1) / n1 + Y0.var(ddof=1) / n0
    sigma_e_sq    = e_hat * (1.0 - e_hat) / n1
    # Cov(Ȳ₁, W̄₁) = sample_Cov(Y,W|Z=1) / n₁
    cov_mat       = np.cov(Y1, W1, ddof=1)
    cov_L_e       = cov_mat[0, 1] / n1

    sigma_CATT_sq = (
        sigma_CITT_sq / e_hat**2
        + CATT**2 * sigma_e_sq / e_hat**2
        - 2.0 * CATT * cov_L_e / e_hat**3
    )
    sigma_L = np.sqrt(max(sigma_CITT_sq, 0.0))
    sigma_U = np.sqrt(max(sigma_CATT_sq, 0.0))

    # ── Imbens-Manski critical value ─────────────────────────────────────────
    gap       = CATT - CITT
    sigma_max = max(sigma_L, sigma_U)
    if gap < 1e-14 or sigma_max < 1e-15:
        c_hat = norm.ppf(1.0 - alpha / 2.0)
    else:
        rng_norm = np.sqrt(n) * gap / sigma_max

        def eq(c):
            return norm.cdf(c + rng_norm) - norm.cdf(-c) - (1.0 - alpha)

        try:
            c_hat = brentq(eq, 0.0, rng_norm + 10.0)
        except Exception:
            c_hat = norm.ppf(1.0 - alpha / 2.0)

    ci_lo = CITT - c_hat * sigma_L / np.sqrt(n)
    ci_hi = CATT + c_hat * sigma_U / np.sqrt(n)

    return dict(CITT=CITT, CATT=CATT, e_hat=e_hat,
                sigma_L=sigma_L, sigma_U=sigma_U,
                ci_lo=ci_lo, ci_hi=ci_hi, width=ci_hi - ci_lo,
                SR_ub=1.0 / e_hat)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3.5.2 Track A: finite-differences MTE + CATE integration
# ─────────────────────────────────────────────────────────────────────────────

def track_a(Y, Z, S, p_levels):
    """
    Finite-differences MTE estimator (Section 3.5.2).

    Within market cell c (S_i = c), P_i = π_c is known and constant.
    μ̂₁(π_c) = Ȳ among (Z_i=1, S_i=c).
    MTE(π̄_c) ≈ [μ̂₁(π_{c+1}) − μ̂₁(π_c)] / (π_{c+1} − π_c).
    CATE = integral + linear extrapolation.
    """
    m1 = Z == 1
    Y1 = Y[m1];  S1 = S[m1]
    C  = len(p_levels)
    mu_hat = np.full(C, np.nan)
    for c in range(C):
        mask = S1 == c
        if mask.sum() >= 2:
            mu_hat[c] = Y1[mask].mean()

    if np.any(np.isnan(mu_hat)):
        return None

    dp      = np.diff(p_levels)
    mte_hat = np.diff(mu_hat) / dp          # finite differences
    p_mid   = 0.5 * (p_levels[:-1] + p_levels[1:])

    p1, pC  = p_levels[0], p_levels[-1]
    cate_hat = (np.sum(mte_hat * dp)
                + mte_hat[0]  * p1
                + mte_hat[-1] * (1.0 - pC))

    return dict(mte_hat=mte_hat, p_mid=p_mid, cate_hat=cate_hat)


# ─────────────────────────────────────────────────────────────────────────────
# DGP specifications and true estimand calculation
# ─────────────────────────────────────────────────────────────────────────────

# ── MTE schedules ────────────────────────────────────────────────────────────

def mte_monotone(U):
    """DGP 1/2/3 and Track A: τ(u) = 2(1−u).  True CATE = 1.0."""
    return 2.0 * (1.0 - U)


def mte_hump(U):
    """DGP-R1: τ(u) = 8u(1−u).  Non-monotone; peaks at u=0.5."""
    return 8.0 * U * (1.0 - U)


def mte_neg_tail(U):
    """DGP-R2: τ(u) = 2 − 4u.  Positive for u<0.5, negative for u>0.5."""
    return 2.0 - 4.0 * U


# ── Assignment rules ─────────────────────────────────────────────────────────

# α for logistic calibrated so E[σ(α+βX)] = 0.30; precomputed below.
_ALPHA_LOGISTIC = None        # logistic DGP 2
_ALPHA_NL       = None        # nonlinear DGP 3 intercept


def _calibrate_alphas(n_mc=1_000_000):
    """Solve for intercepts so that average propensity ≈ 0.30 in each DGP."""
    global _ALPHA_LOGISTIC, _ALPHA_NL
    rng = np.random.default_rng(77)
    X   = rng.standard_normal(n_mc)
    X1  = rng.standard_normal(n_mc)
    X2  = rng.standard_normal(n_mc)

    # DGP 2: E[σ(α + 2X)] = 0.30
    _ALPHA_LOGISTIC = brentq(
        lambda a: sigmoid(a + 2.0 * X).mean() - 0.30, -10.0, 10.0)

    # DGP 3: E[σ(γ₀ + 2X₁² + X₂ − 2X₁X₂)] = 0.30
    _ALPHA_NL = brentq(
        lambda a: sigmoid(a + 2.0*X1**2 + X2 - 2.0*X1*X2).mean() - 0.30,
        -10.0, 10.0)


def assignment_dgp1(n, rng):
    """Constant quota: D = 1{U ≤ 0.30}."""
    U = rng.uniform(0.0, 1.0, n)
    D = (U < 0.30).astype(float)
    return D, U


def assignment_dgp2(n, rng):
    """Logistic score: D = 1{U ≤ σ(α+2X)}, average p ≈ 0.30."""
    X = rng.standard_normal(n)
    U = rng.uniform(0.0, 1.0, n)
    p = sigmoid(_ALPHA_LOGISTIC + 2.0 * X)
    D = (U < p).astype(float)
    return D, U


def assignment_dgp3(n, rng):
    """Nonlinear ML rule: D = 1{U ≤ σ(γ₀+2X₁²+X₂−2X₁X₂)}, avg p ≈ 0.30."""
    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    U  = rng.uniform(0.0, 1.0, n)
    p  = sigmoid(_ALPHA_NL + 2.0*X1**2 + X2 - 2.0*X1*X2)
    D  = (U < p).astype(float)
    return D, U


# ── Compute true estimands by simulation ─────────────────────────────────────

def true_estimands(assign_fn, tau_fn, seed, n_mc=2_000_000):
    """
    Compute true CATE, CATT, CITT, ē, SR via Monte Carlo integration.
    CATT = E[τ(U) | D=1] computed directly as tau[D==1].mean().
    CATE = E[τ(U)] = tau.mean() (since U uniform; exact for monotone and hump).
    """
    rng = np.random.default_rng(seed)
    D, U = assign_fn(n_mc, rng)
    tau  = tau_fn(U)
    CATE = tau.mean()
    e_bar = D.mean()
    if D.sum() == 0:
        return dict(CATE=CATE, CATT=np.nan, CITT=np.nan,
                    e_bar=e_bar, SR=np.nan, SR_ub=np.nan)
    CATT = tau[D == 1].mean()
    CITT = e_bar * CATT
    SR   = CATT / CATE if abs(CATE) > 1e-9 else np.nan
    return dict(CATE=CATE, CATT=CATT, CITT=CITT,
                e_bar=e_bar, SR=SR, SR_ub=1.0 / e_bar)


# ── Sample from a DGP ────────────────────────────────────────────────────────

def sample_dgp(n, assign_fn, tau_fn, rng):
    """Generate one overlay experiment dataset."""
    D, U = assign_fn(n, rng)
    Z    = rng.binomial(1, 0.5, n).astype(float)
    W    = D * Z
    tau  = tau_fn(U)
    Y    = tau * W + rng.standard_normal(n)
    return dict(Y=Y, Z=Z, W=W, D=D, U=U, tau=tau)


# ── Track A DGP: DGP 1 + geographic excluded shifter ─────────────────────────

MARKET_P = np.array([0.20, 0.35, 0.55])   # propensities in 3 markets

def sample_track_a(n, rng):
    """
    DGP 1 (constant propensity per market) + geographic excluded shifter.
    In market s (s=0,1,2), D_i = 1{U_i ≤ π_s}.
    τ(U_i) = 2(1−U_i) as in DGP 1. True MTE(p) = 2(1−p), CATE = 1.0.
    Exclusion restriction: market assignment Z, S independent of U and τ.
    """
    S    = rng.integers(0, 3, n)             # S_i ∈ {0,1,2} market index
    pi_s = MARKET_P[S]                       # propensity in user's market
    U    = rng.uniform(0.0, 1.0, n)
    D    = (U < pi_s).astype(float)
    Z    = rng.binomial(1, 0.5, n).astype(float)
    W    = D * Z
    tau  = 2.0 * (1.0 - U)
    Y    = tau * W + rng.standard_normal(n)
    return dict(Y=Y, Z=Z, W=W, S=S, U=U, tau=tau)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def run_track_b_dgp(assign_fn, tau_fn, tv, label,
                    n_sims=1000, sample_sizes=None, alpha=0.10):
    """Run Track B across sample sizes; report CATT bias/RMSE, CI coverage/width."""
    if sample_sizes is None:
        sample_sizes = [500, 2_000, 10_000, 50_000]

    CATE_t = tv['CATE']
    CATT_t = tv['CATT']
    rows = []

    for n in sample_sizes:
        catt_ests, ci_covers, widths = [], [], []
        for _ in range(n_sims):
            d  = sample_dgp(n, assign_fn, tau_fn, RNG)
            tb = track_b(d['Y'], d['Z'], d['W'], alpha=alpha)
            catt_ests.append(tb['CATT'])
            ci_covers.append(bool(tb['ci_lo'] <= CATE_t <= tb['ci_hi']))
            widths.append(tb['width'])
        ca  = np.array(catt_ests, dtype=float)
        rows.append(dict(
            n           = n,
            CATT_bias   = float(np.nanmean(ca) - CATT_t),
            CATT_RMSE   = float(np.sqrt(np.nanmean((ca - CATT_t)**2))),
            CI_coverage = float(np.mean(ci_covers)),
            CI_width    = float(np.nanmean(widths)),
            SR_ub       = tv['SR_ub'],
        ))

    df = pd.DataFrame(rows)
    print(f"\n{'='*72}")
    print(f"Track B — {label}")
    print(f"  True CATE={CATE_t:.4f}  CATT={CATT_t:.4f}  "
          f"SR={tv['SR']:.3f}  SR_ub(1/ē)={tv['SR_ub']:.3f}")
    print(f"  {'n':>8}  {'CATT bias':>10}  {'CATT RMSE':>10}  "
          f"{'IM CI cov':>10}  {'CI width':>10}")
    for r in df.itertuples():
        print(f"  {r.n:>8,}  {r.CATT_bias:>10.4f}  {r.CATT_RMSE:>10.4f}  "
              f"  {r.CI_coverage:>8.3f}  {r.CI_width:>10.4f}")
    return df


def run_track_a(n_sims=500, sample_sizes=None):
    """Track A: finite-differences MTE and CATE estimation."""
    if sample_sizes is None:
        sample_sizes = [2_000, 10_000, 50_000, 200_000]

    p_mid    = 0.5 * (MARKET_P[:-1] + MARKET_P[1:])  # [0.275, 0.450]
    mte_true = 2.0 * (1.0 - p_mid)                    # [1.45, 1.10]
    cate_true = 1.0                                    # E[2(1-U)] = 1.0

    rows = []
    print(f"\n{'='*72}")
    print("Track A — DGP 1 + geographic excluded shifter")
    print(f"  Markets π_s = {MARKET_P}   True MTE(p) = 2(1-p)")
    print(f"  True MTE at midpoints: {mte_true}")
    print(f"  {'n':>8}  {'MTE bias p=0.275':>16}  {'MTE bias p=0.450':>16}  "
          f"{'CATE RMSE':>12}  {'CATE mean':>12}")

    for n in sample_sizes:
        mte_hats, cate_hats = [], []
        for _ in range(n_sims):
            d   = sample_track_a(n, RNG)
            res = track_a(d['Y'], d['Z'], d['S'], MARKET_P)
            if res is not None:
                mte_hats.append(res['mte_hat'])
                cate_hats.append(res['cate_hat'])

        mte_arr   = np.array(mte_hats)
        cate_arr  = np.array(cate_hats, dtype=float)
        mte_bias  = np.nanmean(mte_arr, axis=0) - mte_true
        cate_rmse = float(np.sqrt(np.nanmean((cate_arr - cate_true)**2)))
        cate_mean = float(np.nanmean(cate_arr))

        rows.append(dict(n=n,
                         MTE_bias_p1=float(mte_bias[0]),
                         MTE_bias_p2=float(mte_bias[1]),
                         CATE_RMSE=cate_rmse,
                         CATE_mean=cate_mean))
        print(f"  {n:>8,}  {mte_bias[0]:>16.4f}  {mte_bias[1]:>16.4f}  "
              f"{cate_rmse:>12.4f}  {cate_mean:>12.4f}")

    return pd.DataFrame(rows), p_mid, mte_true


def run_selection_ratio(n=20_000, n_sims=500):
    """
    Vary treatment rate e ∈ {0.10, 0.30, 0.60} using DGP 1 (constant quota,
    monotone MTE).  Show SR_ub = 1/ê bounds the true SR from above,
    and how CI coverage and width respond to e.
    """
    targets  = [0.10, 0.30, 0.60]
    rows     = []
    print(f"\n{'='*72}")
    print("Selection Ratio Diagnostic — DGP 1 with varying treatment rate e")
    print(f"  True MTE: τ(u)=2(1−u), CATE=1.0 always")
    print(f"  {'e':>6}  {'ē_obs':>8}  {'True SR':>10}  {'SR_ub=1/ē':>12}  "
          f"{'CI cov':>8}  {'CI width':>10}")

    for e_t in targets:
        def _assign(n, rng):
            U = rng.uniform(0.0, 1.0, n)
            D = (U < e_t).astype(float)
            return D, U

        tv   = true_estimands(_assign, mte_monotone, seed=333+int(e_t*100))
        sr_ubs, covers, widths = [], [], []

        for _ in range(n_sims):
            D, U = _assign(n, RNG)
            Z    = RNG.binomial(1, 0.5, n).astype(float)
            W    = D * Z
            tau  = mte_monotone(U)
            Y    = tau * W + RNG.standard_normal(n)
            tb   = track_b(Y, Z, W, alpha=0.10)
            sr_ubs.append(tb['SR_ub'])
            covers.append(bool(tb['ci_lo'] <= tv['CATE'] <= tb['ci_hi']))
            widths.append(tb['width'])

        row = dict(e=e_t,
                   e_obs=float(np.nanmean([1.0/s for s in sr_ubs])),
                   SR_true=float(tv['SR']),
                   SR_ub_mean=float(np.nanmean(sr_ubs)),
                   CI_cov=float(np.mean(covers)),
                   CI_width=float(np.nanmean(widths)))
        rows.append(row)
        print(f"  {e_t:>6.2f}  {row['e_obs']:>8.3f}  {row['SR_true']:>10.3f}  "
              f"{row['SR_ub_mean']:>12.3f}  {row['CI_cov']:>8.3f}  "
              f"{row['CI_width']:>10.4f}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(dfs, tvs, labels,
                 df_r1, tv_r1, df_r2, tv_r2,
                 df_a, p_mid, mte_true_vals,
                 df_sr):

    blue   = '#2166ac'
    red    = '#d6604d'
    green  = '#1a9641'
    orange = '#f4a582'
    purple = '#762a83'
    gray   = '#999999'
    marker = ['o', 's', '^']
    colors_main = [blue, green, purple]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.40)

    # ── Panel A: CATT bias / convergence for DGPs 1-3 ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axhline(0.0, color='k', lw=1.0, ls='--', alpha=0.6)
    for i, (df, label) in enumerate(zip(dfs, labels)):
        ax1.plot(df['n'], df['CATT_bias'], f'{marker[i]}-',
                 color=colors_main[i], lw=1.8, ms=5, label=label)
    ax1.set_xscale('log')
    ax1.set_xlabel('Sample size $n$', fontsize=9)
    ax1.set_ylabel('Bias  (CATT̂ − True CATT)', fontsize=9)
    ax1.set_title('Panel A\nTrack B — CATT Estimator Bias\n'
                  '(three assignment rules, monotone MTE)', fontsize=9)
    ax1.legend(fontsize=7, loc='upper right')

    # ── Panel B: IM CI coverage — main DGPs vs robustness DGPs ───────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axhline(0.90, color='k', lw=1.2, ls='--', label='Nominal 90%')
    for i, (df, label) in enumerate(zip(dfs, labels)):
        ax2.plot(df['n'], df['CI_coverage'], f'{marker[i]}-',
                 color=colors_main[i], lw=1.8, ms=5, label=label)
    ax2.plot(df_r1['n'], df_r1['CI_coverage'], 'D--',
             color=orange, lw=1.5, ms=5, label='DGP-R1 (monotone violated)')
    ax2.plot(df_r2['n'], df_r2['CI_coverage'], 'v:',
             color=red, lw=1.5, ms=5, label='DGP-R2 (nonneg violated)')
    ax2.set_xscale('log')
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_xlabel('Sample size $n$', fontsize=9)
    ax2.set_ylabel('IM CI coverage of true CATE', fontsize=9)
    ax2.set_title('Panel B\nImbens–Manski CI Coverage\n'
                  '(nominal 90%; assumption violations shaded)', fontsize=9)
    ax2.legend(fontsize=7, loc='lower right')

    # ── Panel C: CI width across DGPs ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (df, label) in enumerate(zip(dfs, labels)):
        ax3.plot(df['n'], df['CI_width'], f'{marker[i]}-',
                 color=colors_main[i], lw=1.8, ms=5, label=label)
    ax3.set_xscale('log')
    ax3.set_xlabel('Sample size $n$', fontsize=9)
    ax3.set_ylabel('Mean IM CI width', fontsize=9)
    ax3.set_title('Panel C\nIM CI Width vs. Sample Size\n'
                  '(narrower = more precise bounds)', fontsize=9)
    ax3.legend(fontsize=7)

    # ── Panel D: Track A — true MTE vs estimated ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    p_fine = np.linspace(MARKET_P[0] - 0.05, MARKET_P[-1] + 0.05, 300)
    ax4.plot(p_fine, 2.0*(1.0-p_fine), 'k-', lw=2.0, label='True MTE$(p)=2(1-p)$')
    colors_a = [blue, green, orange, purple]
    for i, row in enumerate(df_a.itertuples()):
        if i >= 4:
            break
        # MTÊ = mte_true + bias; plot with RMSE as approx SD
        mte_est = mte_true_vals + np.array([row.MTE_bias_p1, row.MTE_bias_p2])
        ax4.errorbar(p_mid, mte_est, yerr=row.CATE_RMSE * 0.7,
                     fmt='o', color=colors_a[i], capsize=3, ms=5, lw=1.5,
                     label=f'$n$={row.n:,}')
    ax4.set_xlabel('Propensity level $p$', fontsize=9)
    ax4.set_ylabel('MTE$(p)$', fontsize=9)
    ax4.set_title('Panel D\nTrack A — MTE Finite-Differences\n'
                  '(geographic excluded shifter, DGP 1)', fontsize=9)
    ax4.legend(fontsize=7)

    # ── Panel E: CATE RMSE (Track A) vs bound half-width (Track B) ────────
    ax5 = fig.add_subplot(gs[1, 1])
    ns_a = df_a['n'].values
    ax5.plot(ns_a, df_a['CATE_RMSE'], 'o-', color=blue, lw=1.8, ms=5,
             label='Track A CATE RMSE')
    ax5.plot(dfs[0]['n'], dfs[0]['CI_width'] / 2.0, 's--', color=gray,
             lw=1.4, ms=4, label='Track B CI half-width (DGP 1)')
    ref = df_a['CATE_RMSE'].iloc[0] * np.sqrt(ns_a[0])
    ax5.plot(ns_a, ref / np.sqrt(ns_a), 'k:', lw=1.0, alpha=0.6,
             label='$O(n^{-1/2})$ reference')
    ax5.set_xscale('log');  ax5.set_yscale('log')
    ax5.set_xlabel('Sample size $n$', fontsize=9)
    ax5.set_ylabel('RMSE / CI half-width', fontsize=9)
    ax5.set_title('Panel E\nTrack A RMSE vs Track B Width\n'
                  '($\\sqrt{n}$ convergence; Track A wins at large $n$)', fontsize=9)
    ax5.legend(fontsize=7)

    # ── Panel F: Selection Ratio diagnostic ───────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    e_vals = df_sr['e'].values
    ax6.plot(e_vals, df_sr['SR_ub_mean'], 'o-', color=red, lw=1.8, ms=7,
             label='$1/\\hat{e}$ (SR upper bound, obs.)')
    ax6.plot(e_vals, df_sr['SR_true'], 's--', color=blue, lw=1.8, ms=7,
             label='True Selection Ratio')
    ax6.axhline(1.0, color='k', lw=0.8, ls=':', alpha=0.6,
                label='SR = 1 (no cherry-picking)')
    ax6.set_xlabel('Treatment rate $e$', fontsize=9)
    ax6.set_ylabel('Selection Ratio', fontsize=9)
    ax6.set_title('Panel F\nSelection Ratio Diagnostic\n'
                  '($1/\\hat{e}$ is sharp upper bound on SR)', fontsize=9)
    ax6.legend(fontsize=7)

    fig.suptitle(
        '"Measuring Causal Effects under Opaque Targeting" — Monte Carlo Results\n'
        'DGP 1: Constant quota  |  DGP 2: Logistic score  |  DGP 3: Nonlinear ML  '
        '|  R1: Monotone violated  |  R2: Nonneg violated',
        fontsize=10, fontweight='bold'
    )

    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'mc_results.{ext}', bbox_inches='tight', dpi=200)
    print(f"\nFigure saved: {OUT}/mc_results.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*72)
    print("MONTE CARLO — OVERLAY EXPERIMENT PAPER  (Track A + Track B)")
    print("="*72 + "\n")

    # Calibrate targeting-rule intercepts
    print("Calibrating logistic and nonlinear assignment rules to ē = 0.30 ...")
    _calibrate_alphas()
    print(f"  DGP2 logistic intercept α = {_ALPHA_LOGISTIC:.3f}")
    print(f"  DGP3 nonlinear intercept γ₀ = {_ALPHA_NL:.3f}\n")

    # True estimands
    print("Computing true estimands (2M-draw MC) ...")
    assign_fns = [assignment_dgp1, assignment_dgp2, assignment_dgp3]
    labels_short = ['DGP 1 (constant quota)',
                    'DGP 2 (logistic score)',
                    'DGP 3 (nonlinear ML)']

    tvs = [true_estimands(fn, mte_monotone, seed=s)
           for fn, s in zip(assign_fns, [10, 20, 30])]

    def _assign_const(e_fixed):
        def f(n, rng):
            U = rng.uniform(0.0, 1.0, n)
            D = (U < e_fixed).astype(float)
            return D, U
        return f

    tv_r1 = true_estimands(_assign_const(0.30), mte_hump,     seed=40)
    tv_r2 = true_estimands(_assign_const(0.30), mte_neg_tail, seed=50)

    for tv, lab in zip(tvs, labels_short):
        print(f"  {lab}")
        print(f"    CATE={tv['CATE']:.4f}  CATT={tv['CATT']:.4f}  "
              f"CITT={tv['CITT']:.4f}  ē={tv['e_bar']:.3f}  "
              f"SR={tv['SR']:.3f}  SR_ub={tv['SR_ub']:.3f}")
    print(f"  DGP-R1 (hump MTE, e=0.30): CATE={tv_r1['CATE']:.4f}  "
          f"CATT={tv_r1['CATT']:.4f}  "
          f"(CATE > CATT → upper bound fails)")
    print(f"  DGP-R2 (neg tail,  e=0.30): CATE={tv_r2['CATE']:.4f}  "
          f"CITT={tv_r2['CITT']:.4f}  "
          f"(CATE < CITT → lower bound fails)\n")

    # ── Track B experiments ──────────────────────────────────────────────────
    SIZES_B = [500, 2_000, 10_000, 50_000]
    N_SIMS  = 1000

    dfs = [
        run_track_b_dgp(fn, mte_monotone, tv, lab,
                        n_sims=N_SIMS, sample_sizes=SIZES_B, alpha=0.10)
        for fn, tv, lab in zip(assign_fns, tvs, labels_short)
    ]

    df_r1 = run_track_b_dgp(_assign_const(0.30), mte_hump, tv_r1,
                             'DGP-R1 (hump MTE, monotone selection violated)',
                             n_sims=N_SIMS, sample_sizes=SIZES_B, alpha=0.10)

    df_r2 = run_track_b_dgp(_assign_const(0.30), mte_neg_tail, tv_r2,
                             'DGP-R2 (negative-tail MTE, nonneg violated)',
                             n_sims=N_SIMS, sample_sizes=SIZES_B, alpha=0.10)

    # ── Track A ──────────────────────────────────────────────────────────────
    SIZES_A = [2_000, 10_000, 50_000, 200_000]
    df_a, p_mid, mte_true_vals = run_track_a(n_sims=500, sample_sizes=SIZES_A)

    # ── Selection Ratio diagnostic ───────────────────────────────────────────
    df_sr = run_selection_ratio(n=20_000, n_sims=500)

    # ── Save tables ──────────────────────────────────────────────────────────
    for df, tag in zip(dfs, ['dgp1', 'dgp2', 'dgp3']):
        df.to_csv(OUT / f'track_b_{tag}.csv', index=False)
    df_r1.to_csv(OUT / 'track_b_r1_hump.csv', index=False)
    df_r2.to_csv(OUT / 'track_b_r2_negtail.csv', index=False)
    df_a.to_csv(OUT  / 'track_a_mte.csv', index=False)
    df_sr.to_csv(OUT / 'selection_ratio.csv', index=False)

    # ── Figures ──────────────────────────────────────────────────────────────
    make_figures(dfs, tvs, labels_short,
                 df_r1, tv_r1, df_r2, tv_r2,
                 df_a, p_mid, mte_true_vals, df_sr)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("SUMMARY — Track B IM CI coverage at n=50,000 (nominal 90%)")
    for df, lab in zip(dfs, labels_short):
        cov = df['CI_coverage'].iloc[-1]
        print(f"  {lab:<35s}  {cov:.3f}")
    print(f"  {'DGP-R1 (monotone violated)':<35s}  {df_r1['CI_coverage'].iloc[-1]:.3f}")
    print(f"  {'DGP-R2 (nonneg violated)':<35s}  {df_r2['CI_coverage'].iloc[-1]:.3f}")
    row50 = df_a[df_a['n'] == 50_000].squeeze()
    if hasattr(row50, 'CATE_RMSE'):
        print(f"\nTrack A CATE RMSE at n=50,000: {row50.CATE_RMSE:.4f}"
              f"  (true CATE = 1.0000)")
    print("\nAll simulations complete.")


if __name__ == '__main__':
    main()
