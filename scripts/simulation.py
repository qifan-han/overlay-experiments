# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "scipy>=1.12",
#   "matplotlib>=3.8",
#   "pandas>=2.2",
#   "tqdm>=4.66",
# ]
# ///
"""
Monte Carlo Simulations: "Measuring Causal Effects under Opaque Targeting"

Three data-generating processes model concrete real-world scenarios where a
platform's targeting algorithm is opaque to the experimenter. Estimators are
applied without knowledge of the DGP; results are evaluated against true
estimands computed by large-sample Monte Carlo integration.

────────────────────────────────────────────────────────────────────────────
DGP 1 — Meta/Facebook Advertiser (Simple Logistic Targeting)
────────────────────────────────────────────────────────────────────────────
A DTC brand measures conversion lift on Meta. Meta's ad auction scores users
on purchase intent X ~ U[0,1] using a logistic model: targeting probability
p(X) = sigmoid(-2 + 4X). High-intent users (X→1) face p ≈ 88%, low-intent
(X→0) face p ≈ 12%. Two forms of selection operate simultaneously:

  - Selection into treatment on observables: high-X users have higher
    baseline conversions (Y₀ = 0.5·X + ε), so OLS naively overestimates
    the ad effect by attributing baseline advantage to the campaign.
  - Selection on gains: high-X users also respond more strongly to ads
    (τ(X) = 1 + 2X), so ATT > ATE — the platform's targeting achieves
    better-than-average lift.

The brand runs a budget-level randomization Z ~ Bernoulli(0.5): Z=1 means
the campaign is live for user i; Z=0 means it is paused. They observe (Y, Z,
W=D·Z) but NOT Meta's propensity p(X) or individual targeting status D.

Three estimators are compared:
  (a) Naive OLS: regresses Y on W — biased by the baseline-selection channel.
  (b) Plain ITT: E[Y|Z=1] − E[Y|Z=0] — underestimates ATT by factor p̄ ≈ 0.5.
  (c) Overlay ATT: ITT̂ / p̂ — correctly identifies ATT (Proposition 1).

────────────────────────────────────────────────────────────────────────────
DGP 2 — TikTok External Advertiser (Nonlinear ML Targeting)
────────────────────────────────────────────────────────────────────────────
A music label tests a promotional campaign on TikTok. TikTok's ranking engine
determines targeting via a nonlinear interaction of user age (X₁) and content
affinity (X₂):

    p(X₁, X₂) = sigmoid(−1 + 3·X₁² + 2·X₂ − 4·X₁·X₂)

Young, high-affinity users get disproportionately high targeting probability;
the negative interaction X₁·X₂ suppresses it for older high-affinity users —
a pattern typical of neural-network ranking models. The label cannot
replicate or query TikTok's model. They apply the same overlay design (Z=1:
campaign shown; Z=0: withheld). The estimator is applied with no knowledge
of the functional form of p(·,·). This experiment tests whether the
identification result (Proposition 1) is robust to arbitrary nonlinear and
non-separable targeting rules.

────────────────────────────────────────────────────────────────────────────
DGP 3 — Platform O: External Advertiser Alongside Ye et al.'s m=3 Setup
────────────────────────────────────────────────────────────────────────────
Platform O (a TikTok-like platform, as studied by Ye et al. 2025) runs m=3
concurrent A/B tests on its recommendation algorithm. The outcome Y follows
Ye et al.'s generalized sigmoid form II (their equation 6):

    E[Y|X, T] = θ₄(X) / (1 + exp(−(θ₀(X) + θ₁(X)·T₁ + θ₂(X)·T₂ + θ₃(X)·T₃)))

An external brand (Firm A) simultaneously runs an overlay experiment. Firm A
has no access to Platform O's internal assignment rule v(t|x) — the precise
information that Ye et al.'s DeDL requires and that external advertisers
cannot obtain. Firm B (a second external advertiser) competes for the same
users in a real-time bidding auction, creating competition-induced correlation
between Firm A's targeting probability p_A(X₁, X₂) and Firm B's activity.

From Firm A's perspective:
  - D_A = T₁ (whether Platform O serves Firm A's ad, with p_A depending on X₁
    and on X₂ through auction competition with Firm B — both opaque to Firm A)
  - Z_A ~ Bernoulli(0.5) (Firm A's own overlay gate)
  - W_A = D_A · Z_A (realized ad exposure)
  - Y_A = own outcome (purchases), affected only by W_A in this DGP

Firm A's true ATT_A = E[τ_A(X₁) | D_A=1] is computed numerically. The
overlay estimator ITT_A/p̂_A identifies ATT_A without knowing v(t|x),
p_A(·,·), or anything about Firm B. This directly contrasts with Ye et al.'s
internal-platform DeDL, which assumes v(t|x) is known.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ─────────────────────────────────────────────────────────────────────────────
# DGP 1: Meta/Facebook logistic targeting
# ─────────────────────────────────────────────────────────────────────────────

def dgp1(n, rng):
    X = rng.uniform(0, 1, n)
    U = rng.uniform(0, 1, n)                     # latent resistance to targeting
    p = sigmoid(-2.0 + 4.0 * X)                  # Meta's opaque logistic rule
    D = (U < p).astype(float)                     # targeted if latent resistance low
    Z = rng.binomial(1, 0.5, n).astype(float)     # brand's overlay gate
    W = D * Z                                      # realized ad exposure
    tau = 1.0 + 2.0 * X                           # treatment effect (selection on gains)
    Y0 = 0.5 * X + rng.normal(0, 1, n)           # baseline correlated with X (selection bias)
    Y = Y0 + tau * W
    return dict(X=X, U=U, p=p, D=D, Z=Z, W=W, Y=Y, tau=tau)


def true_att_dgp1(n_mc=2_000_000):
    """
    ATT = E[tau(X) | D=1] = E[tau(X)·p(X)] / E[p(X)]
    (since D=1{U<p(X)} and U⊥X, so P(D=1|X)=p(X))
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, n_mc)
    p = sigmoid(-2.0 + 4.0 * X)
    tau = 1.0 + 2.0 * X
    return float(np.sum(tau * p) / np.sum(p))


TRUE_ATE_DGP1 = 2.0   # E[1+2X] for X~U[0,1] — closed form


# ─────────────────────────────────────────────────────────────────────────────
# DGP 2: TikTok nonlinear ML targeting
# ─────────────────────────────────────────────────────────────────────────────

def dgp2(n, rng):
    X1 = rng.uniform(0, 1, n)
    X2 = rng.uniform(0, 1, n)
    U  = rng.uniform(0, 1, n)
    p  = sigmoid(-1.0 + 3.0 * X1**2 + 2.0 * X2 - 4.0 * X1 * X2)
    D  = (U < p).astype(float)
    Z  = rng.binomial(1, 0.5, n).astype(float)
    W  = D * Z
    tau = 1.0 + X1 + X2
    Y0  = 0.3 * X1 + 0.3 * X2 + rng.normal(0, 1, n)
    Y   = Y0 + tau * W
    return dict(X1=X1, X2=X2, U=U, p=p, D=D, Z=Z, W=W, Y=Y, tau=tau)


def true_att_dgp2(n_mc=2_000_000):
    rng = np.random.default_rng(1)
    X1 = rng.uniform(0, 1, n_mc)
    X2 = rng.uniform(0, 1, n_mc)
    p   = sigmoid(-1.0 + 3.0 * X1**2 + 2.0 * X2 - 4.0 * X1 * X2)
    tau = 1.0 + X1 + X2
    return float(np.sum(tau * p) / np.sum(p))


# ─────────────────────────────────────────────────────────────────────────────
# DGP 3: Platform O (Ye et al. setup) + external advertiser with auction
# ─────────────────────────────────────────────────────────────────────────────

def ye_outcome(X1, X2, T1, T2, T3):
    """
    Ye et al.'s generalized sigmoid form II (their equation 6) with:
        theta_0(X) = -1.0 + 0.8·X1 + 0.5·X2     (baseline)
        theta_1(X) = 1.0 + 1.5·X1                 (T1 effect, heterogeneous)
        theta_2(X) = 0.8 + 0.5·X2                 (T2 effect)
        theta_3(X) = 0.6                           (T3 effect, homogeneous)
        theta_4(X) = 5.0                           (scale)
    """
    th0 = -1.0 + 0.8 * X1 + 0.5 * X2
    th1 =  1.0 + 1.5 * X1
    th2 =  0.8 + 0.5 * X2
    th3 =  0.6
    th4 =  5.0
    return th4 / (1.0 + np.exp(-(th0 + th1 * T1 + th2 * T2 + th3 * T3)))


def dgp3(n, rng):
    X1 = rng.uniform(0, 1, n)    # purchase intent (Firm A's primary covariate)
    X2 = rng.uniform(0, 1, n)    # brand affinity (Firm B's primary covariate)

    # Platform O's three experiments — randomized, but targeting is based on X1
    # Firm A's targeting probability: positively influenced by X1,
    # negatively by X2 (auction competition from Firm B)
    pA = sigmoid(2.0 * X1 - 1.5 * X2)           # Firm A wins auction
    pB = sigmoid(2.0 * X2 - 1.0 * X1 + 0.3)     # Firm B wins auction

    # Platform's other two experiments (T2, T3) are uniformly randomized
    # (analogous to Ye et al.'s orthogonal internal experiments)
    T2 = rng.binomial(1, 0.6, n).astype(float)
    T3 = rng.binomial(1, 0.6, n).astype(float)

    UA = rng.uniform(0, 1, n)
    UB = rng.uniform(0, 1, n)
    DA = (UA < pA).astype(float)    # Firm A's ad shown (platform decision)
    DB = (UB < pB).astype(float)    # Firm B's ad shown

    ZA = rng.binomial(1, 0.5, n).astype(float)   # Firm A's overlay gate
    ZB = rng.binomial(1, 0.5, n).astype(float)   # Firm B's overlay gate
    WA = DA * ZA
    WB = DB * ZB

    # Platform O's outcome (screen time) — Ye et al.'s generalized sigmoid
    # T1 for Platform O = DA (Firm A's experiment is one of Platform O's three)
    Y_platform = ye_outcome(X1, X2, DA, T2, T3) + rng.normal(0, 0.5, n)

    # Firm A's own outcome (purchases): depends only on WA, not platform's Y
    tauA = 1.0 + 1.5 * X1
    YA   = 0.3 * X1 + rng.normal(0, 1, n) + tauA * WA

    # Firm B's own outcome (brand recall): depends only on WB
    tauB = 0.5 + 1.5 * X2
    YB   = 0.3 * X2 + rng.normal(0, 1, n) + tauB * WB

    return dict(X1=X1, X2=X2, pA=pA, pB=pB,
                DA=DA, DB=DB, T2=T2, T3=T3,
                ZA=ZA, ZB=ZB, WA=WA, WB=WB,
                tauA=tauA, tauB=tauB,
                YA=YA, YB=YB, Y_platform=Y_platform)


def true_att_dgp3(n_mc=2_000_000):
    rng = np.random.default_rng(2)
    X1 = rng.uniform(0, 1, n_mc)
    X2 = rng.uniform(0, 1, n_mc)
    pA   = sigmoid(2.0 * X1 - 1.5 * X2)
    pB   = sigmoid(2.0 * X2 - 1.0 * X1 + 0.3)
    tauA = 1.0 + 1.5 * X1
    tauB = 0.5 + 1.5 * X2
    att_A = float(np.sum(tauA * pA) / np.sum(pA))
    att_B = float(np.sum(tauB * pB) / np.sum(pB))
    return att_A, att_B


# ─────────────────────────────────────────────────────────────────────────────
# Estimators
# ─────────────────────────────────────────────────────────────────────────────

def overlay_att(Y, Z, W):
    """Proposition 1 estimator: ATT̂ = ITT̂ / p̂ = (Ȳ[Z=1]−Ȳ[Z=0]) / P̂(W=1|Z=1)"""
    m1, m0 = (Z == 1), (Z == 0)
    ITT   = Y[m1].mean() - Y[m0].mean()
    p_hat = W[m1].mean()
    return ITT / p_hat if p_hat > 1e-9 else np.nan


def naive_ols(Y, W):
    """
    Naive OLS: β̂ = Cov(Y,W)/Var(W).
    Biased because D correlates with Y₀ through X (omitted variable).
    """
    Wdm  = W - W.mean()
    denom = (Wdm**2).sum()
    return float(np.dot(Wdm, Y) / denom) if denom > 1e-12 else np.nan


def plain_itt(Y, Z):
    """
    Plain ITT: Ȳ[Z=1] − Ȳ[Z=0].
    Underestimates ATT by factor p̄ (the average targeting probability).
    """
    return float(Y[Z == 1].mean() - Y[Z == 0].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: DGP1 — logistic targeting, estimator comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_exp1(ATT_true, n_sims=500, sample_sizes=None):
    if sample_sizes is None:
        sample_sizes = [500, 2_000, 10_000, 50_000]

    print("=" * 68)
    print("EXPERIMENT 1  DGP1: Meta/Facebook Logistic Targeting")
    print("Proposition 1 — overlay identifies ATT; naive estimators fail")
    print("=" * 68)
    print(f"  True ATT = {ATT_true:.4f}    True ATE = {TRUE_ATE_DGP1:.4f}"
          f"    Selection ratio ATT/ATE = {ATT_true/TRUE_ATE_DGP1:.3f}")
    print()

    rows = []
    for n in sample_sizes:
        ov, ols, itt = [], [], []
        for _ in range(n_sims):
            d = dgp1(n, RNG)
            ov.append(overlay_att(d['Y'], d['Z'], d['W']))
            ols.append(naive_ols(d['Y'], d['W']))
            itt.append(plain_itt(d['Y'], d['Z']))
        ov, ols, itt = np.array(ov), np.array(ols), np.array(itt)
        rows.append({
            'n': n,
            'Overlay_mean': ov.mean(),
            'Overlay_bias': ov.mean() - ATT_true,
            'Overlay_RMSE': np.sqrt(((ov - ATT_true)**2).mean()),
            'OLS_mean':     ols.mean(),
            'OLS_bias':     ols.mean() - ATT_true,
            'ITT_mean':     itt.mean(),
            'ITT_bias':     itt.mean() - ATT_true,
        })
        print(f"  n={n:6d}:  Overlay={ov.mean():.4f} (bias={ov.mean()-ATT_true:+.4f}) | "
              f"OLS={ols.mean():.4f} (bias={ols.mean()-ATT_true:+.4f}) | "
              f"ITT={itt.mean():.4f} (bias={itt.mean()-ATT_true:+.4f})")
    print()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: DGP2 — nonlinear ML targeting, robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_exp2(ATT_true, n_sims=400, sample_sizes=None):
    if sample_sizes is None:
        sample_sizes = [2_000, 10_000, 50_000]

    print("=" * 68)
    print("EXPERIMENT 2  DGP2: TikTok Nonlinear ML Targeting")
    print("Proposition 1 robustness — arbitrary nonlinear opaque rule")
    print("=" * 68)
    print(f"  True ATT = {ATT_true:.4f}")
    print()

    rows = []
    for n in sample_sizes:
        ov, ols, itt = [], [], []
        for _ in range(n_sims):
            d = dgp2(n, RNG)
            ov.append(overlay_att(d['Y'], d['Z'], d['W']))
            ols.append(naive_ols(d['Y'], d['W']))
            itt.append(plain_itt(d['Y'], d['Z']))
        ov, ols, itt = np.array(ov), np.array(ols), np.array(itt)
        rows.append({
            'n': n,
            'Overlay_mean': ov.mean(),
            'Overlay_bias': ov.mean() - ATT_true,
            'Overlay_RMSE': np.sqrt(((ov - ATT_true)**2).mean()),
            'OLS_mean':     ols.mean(),
            'OLS_bias':     ols.mean() - ATT_true,
            'ITT_mean':     itt.mean(),
            'ITT_bias':     itt.mean() - ATT_true,
        })
        print(f"  n={n:6d}:  Overlay={ov.mean():.4f} (bias={ov.mean()-ATT_true:+.4f}) | "
              f"OLS={ols.mean():.4f} (bias={ols.mean()-ATT_true:+.4f}) | "
              f"ITT={itt.mean():.4f} (bias={itt.mean()-ATT_true:+.4f})")
    print()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: DGP3 — Platform O / multi-firm simultaneous experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_exp3(ATT_A_true, ATT_B_true, n_sims=400, sample_sizes=None):
    if sample_sizes is None:
        sample_sizes = [500, 2_000, 10_000, 50_000]

    print("=" * 68)
    print("EXPERIMENT 3  DGP3: Platform O — Two Simultaneous Overlay Experiments")
    print("Proposition 1 in the multi-firm setting (Ye et al.'s context)")
    print("=" * 68)
    print(f"  True ATT_A (electronics) = {ATT_A_true:.4f}")
    print(f"  True ATT_B (fashion)     = {ATT_B_true:.4f}")
    print()

    rows_A, rows_B = [], []
    for n in sample_sizes:
        ovA, ovB = [], []
        for _ in range(n_sims):
            d = dgp3(n, RNG)
            ovA.append(overlay_att(d['YA'], d['ZA'], d['WA']))
            ovB.append(overlay_att(d['YB'], d['ZB'], d['WB']))
        ovA, ovB = np.array(ovA), np.array(ovB)
        rows_A.append({'n': n,
                       'ATT_A_mean': ovA.mean(),
                       'ATT_A_bias': ovA.mean() - ATT_A_true,
                       'ATT_A_RMSE': np.sqrt(((ovA - ATT_A_true)**2).mean())})
        rows_B.append({'n': n,
                       'ATT_B_mean': ovB.mean(),
                       'ATT_B_bias': ovB.mean() - ATT_B_true,
                       'ATT_B_RMSE': np.sqrt(((ovB - ATT_B_true)**2).mean())})
        print(f"  n={n:6d}:  "
              f"ATT_A={ovA.mean():.4f} (bias={ovA.mean()-ATT_A_true:+.4f}) | "
              f"ATT_B={ovB.mean():.4f} (bias={ovB.mean()-ATT_B_true:+.4f})")
    print()
    return pd.DataFrame(rows_A), pd.DataFrame(rows_B)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: MTE identification with excluded shifter — Theorem 1
#
# Real-world context (DGP1 extended):
# Meta allows the brand to test different daily budget levels S, which shift
# the effective targeting intensity.  Higher S → Meta targets more users.
# S is randomly assigned across geographic markets (excluded shifter:
# S ⊥ (Y(0), Y(1))).  Treatment effect heterogeneity: tau(U) = 2(1−U) —
# marginal users (high U) respond less, consistent with decreasing returns.
# Theorem 1: MTE(p) = ∂/∂p E[Y|Z=1, P=p] = 2(1−p), identified nonparametrically.
# ─────────────────────────────────────────────────────────────────────────────

def dgp_mte(n, s_vals, rng):
    """
    S_i ~ Uniform over s_vals (campaign budget levels / geographic markets).
    P_i = S_i  (average propensity equals budget level — clean identification).
    tau(U_i) = 2(1−U_i): monotone decreasing in latent resistance.
    True MTE(p) = 2(1−p).
    """
    idx = rng.integers(0, len(s_vals), n)
    S   = s_vals[idx]
    P   = S                                    # propensity = budget level
    U   = rng.uniform(0, 1, n)
    D   = (U < P).astype(float)
    Z   = rng.binomial(1, 0.5, n).astype(float)
    W   = D * Z
    tau = 2.0 * (1.0 - U)
    Y   = rng.normal(0, 1, n) + tau * W
    return dict(S=S, P=P, U=U, D=D, Z=Z, W=W, Y=Y, tau=tau)


def estimate_mte(data, p_grid, bw=0.08):
    """Local linear derivative: MTE(p) = d/dp E[Y|Z=1, P=p]."""
    m1    = data['Z'] == 1
    P1, Y1 = data['P'][m1], data['Y'][m1]
    mte_hat = np.zeros(len(p_grid))
    for j, p0 in enumerate(p_grid):
        K  = np.exp(-0.5 * ((P1 - p0) / bw)**2) / (bw * np.sqrt(2 * np.pi))
        dp = P1 - p0
        s0, s1, s2 = K.sum(), (K * dp).sum(), (K * dp**2).sum()
        t0, t1     = (K * Y1).sum(), (K * dp * Y1).sum()
        denom = s0 * s2 - s1**2
        mte_hat[j] = (s0 * t1 - s1 * t0) / denom if abs(denom) > 1e-12 else np.nan
    return mte_hat


def run_exp4(n=50_000, n_sims=200):
    s_vals   = np.array([0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85])
    p_grid   = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    mte_true = 2.0 * (1.0 - p_grid)

    print("=" * 68)
    print("EXPERIMENT 4  MTE Identification with Excluded Shifter")
    print("Theorem 1 — budget-level S as geographic-market instrument")
    print("=" * 68)
    print(f"  S ∈ {s_vals};  True MTE(p) = 2(1−p)")
    print()

    mte_hats = np.zeros((n_sims, len(p_grid)))
    for sim in range(n_sims):
        d = dgp_mte(n, s_vals, RNG)
        mte_hats[sim] = estimate_mte(d, p_grid)

    mean_hat = np.nanmean(mte_hats, axis=0)
    sd_hat   = np.nanstd(mte_hats,  axis=0)
    bias     = mean_hat - mte_true
    rmse     = np.sqrt(bias**2 + sd_hat**2)

    rows = [{'p': p_grid[j], 'True_MTE': mte_true[j], 'Estimated': mean_hat[j],
             'Bias': bias[j], 'SD': sd_hat[j], 'RMSE': rmse[j]}
            for j in range(len(p_grid))]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format='%.4f'))
    print()
    return df, p_grid, mte_true, mean_hat, sd_hat


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: Sharp bounds [ITT, ATT] — Theorem 2
#
# Real-world context: In the Meta advertiser setting (DGP1), the platform
# targets 40% of users on average.  We know ATE ∈ [ITT, ATT] but ATE is
# not point-identified without a shifter.  Three MTE schedules test sharpness:
#   A — Only targeted users respond: ATE = ITT (lower bound tight)
#   B — All users respond equally:  ATE = ATT (upper bound tight)
#   C — Heterogeneous response:     ATE ∈ interior (realistic case)
# ─────────────────────────────────────────────────────────────────────────────

def run_exp5(n=20_000, n_sims=500):
    p_bar = 0.4   # representative average targeting rate (DGP1 average p̄ ≈ 0.5;
                  # we use 0.4 here for exact alignment with paper's worked example)

    def sample_bounds(n, tau_func, rng):
        U = rng.uniform(0, 1, n)
        D = (U < p_bar).astype(float)
        Z = rng.binomial(1, 0.5, n).astype(float)
        W = D * Z
        Y = rng.normal(0, 1, n) + tau_func(U) * W
        return dict(U=U, D=D, Z=Z, W=W, Y=Y)

    # DGP-A: only targeted (low-U) users respond → ATE = ITT
    tau_A = lambda U: 1.6 * (U <= p_bar).astype(float)
    # DGP-B: constant effect → ATE = ATT
    tau_B = lambda U: np.full_like(U, 1.6)
    # DGP-C: decreasing MTE → ATE interior
    tau_C = lambda U: 2.0 * (1.0 - U)

    dgps = {
        'A: only targeted respond (ATE=ITT)': (tau_A, p_bar * 1.6, p_bar * 1.6, 1.6),
        'B: uniform response (ATE=ATT)':      (tau_B,         1.6, p_bar * 1.6, 1.6),
        'C: decreasing MTE (ATE interior)':   (tau_C,         1.0, p_bar * 1.6, 1.6),
    }

    print("=" * 68)
    print("EXPERIMENT 5  Sharp Bounds [ITT, ATT]")
    print("Theorem 2 — three DGPs test lower endpoint, upper endpoint, interior")
    print("=" * 68)
    print(f"  Fixed p̄={p_bar}: true ITT={p_bar*1.6:.4f}, true ATT=1.6000")
    print()

    results = {}
    for label, (tau_func, ATE_true, ITT_true, ATT_true) in dgps.items():
        lbs, ubs, covered = [], [], []
        for _ in range(n_sims):
            d    = sample_bounds(n, tau_func, RNG)
            lb   = plain_itt(d['Y'], d['Z'])
            ub   = overlay_att(d['Y'], d['Z'], d['W'])
            lbs.append(lb); ubs.append(ub)
            covered.append(lb <= ATE_true <= ub)
        lbs, ubs = np.array(lbs), np.array(ubs)
        results[label] = {
            'ATE_true': ATE_true, 'ITT_true': ITT_true, 'ATT_true': ATT_true,
            'LB_mean': lbs.mean(), 'UB_mean': ubs.mean(),
            'Coverage': np.mean(covered),
        }
        print(f"  {label}")
        print(f"    True:  ITT={ITT_true:.4f}, ATE={ATE_true:.4f}, ATT={ATT_true:.4f}")
        print(f"    Est:   LB={lbs.mean():.4f}, UB={ubs.mean():.4f},  Coverage={np.mean(covered)*100:.1f}%")
        print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(df1, ATT1,
                 df2, ATT2,
                 df3A, df3B, ATT_A, ATT_B,
                 df4, p_grid, mte_true, mte_mean, mte_sd,
                 exp5_results):

    fig = plt.figure(figsize=(17, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    blue   = '#2166ac'
    red    = '#d6604d'
    orange = '#f4a582'

    # ── Panel A: DGP1 estimator comparison ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ns  = df1['n'].values
    ax1.axhline(ATT1,         color='k',   lw=1.4, ls='--',
                label=f'True ATT = {ATT1:.3f}')
    ax1.axhline(TRUE_ATE_DGP1, color='gray', lw=0.8, ls=':',
                label=f'True ATE = {TRUE_ATE_DGP1:.1f}')
    ax1.plot(ns, df1['Overlay_mean'], 'o-', color=blue,   lw=1.8, ms=5,
             label='Overlay ATT̂  (correct)')
    ax1.plot(ns, df1['OLS_mean'],     's-', color=red,    lw=1.8, ms=5,
             label='Naive OLS  (biased ↑)')
    ax1.plot(ns, df1['ITT_mean'],     '^-', color=orange, lw=1.8, ms=5,
             label='Plain ITT  (biased ↓)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Sample size n', fontsize=9)
    ax1.set_ylabel('Estimated effect', fontsize=9)
    ax1.set_title('Panel A\nDGP1: Meta Logistic Targeting\n(Proposition 1)', fontsize=9)
    ax1.legend(fontsize=7)

    # ── Panel B: DGP2 estimator comparison ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ns2 = df2['n'].values
    ax2.axhline(ATT2, color='k', lw=1.4, ls='--',
                label=f'True ATT = {ATT2:.3f}')
    ax2.plot(ns2, df2['Overlay_mean'], 'o-', color=blue,   lw=1.8, ms=5,
             label='Overlay ATT̂  (correct)')
    ax2.plot(ns2, df2['OLS_mean'],     's-', color=red,    lw=1.8, ms=5,
             label='Naive OLS  (biased ↑)')
    ax2.plot(ns2, df2['ITT_mean'],     '^-', color=orange, lw=1.8, ms=5,
             label='Plain ITT  (biased ↓)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Sample size n', fontsize=9)
    ax2.set_ylabel('Estimated effect', fontsize=9)
    ax2.set_title('Panel B\nDGP2: TikTok Nonlinear ML Targeting\n(Proposition 1 robustness)', fontsize=9)
    ax2.legend(fontsize=7)

    # ── Panel C: DGP3 — two-firm convergence ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ns3 = df3A['n'].values
    ax3.axhline(ATT_A, color=blue,   lw=1.2, ls='--',
                label=f'True ATT_A = {ATT_A:.3f}')
    ax3.axhline(ATT_B, color=orange, lw=1.2, ls='--',
                label=f'True ATT_B = {ATT_B:.3f}')
    ax3.plot(ns3, df3A['ATT_A_mean'], 'o-', color=blue,   lw=1.8, ms=5,
             label='Firm A overlay')
    ax3.plot(ns3, df3B['ATT_B_mean'], 's-', color=orange, lw=1.8, ms=5,
             label='Firm B overlay')
    ax3.set_xscale('log')
    ax3.set_xlabel('Sample size n', fontsize=9)
    ax3.set_ylabel('Estimated ATT', fontsize=9)
    ax3.set_title('Panel C\nDGP3: Platform O — Two Firms\n(Proposition 1, multi-firm)', fontsize=9)
    ax3.legend(fontsize=7)

    # ── Panel D: MTE true vs estimated ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    p_fine = np.linspace(0.1, 0.9, 300)
    ax4.plot(p_fine, 2*(1-p_fine), 'k-', lw=2, label='True MTE(p) = 2(1−p)')
    ax4.errorbar(p_grid, mte_mean, yerr=1.96*mte_sd, fmt='o',
                 color=red, capsize=3, lw=1.5, ms=5,
                 label='MC mean ± 1.96·SD  (n=50k, 200 reps)')
    ax4.set_xlabel('Propensity score p', fontsize=9)
    ax4.set_ylabel('MTE(p)', fontsize=9)
    ax4.set_title('Panel D\nMTE Identification via Budget Shifter\n(Theorem 1)', fontsize=9)
    ax4.legend(fontsize=7)

    # ── Panel E: Sharp bounds ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    clrs = ['#1a9641', '#2166ac', '#d73027']
    for j, (label, res) in enumerate(exp5_results.items()):
        x  = j + 1
        lb, ub, ate = res['LB_mean'], res['UB_mean'], res['ATE_true']
        ax5.plot([x, x], [lb, ub], '-', color=clrs[j], lw=9, alpha=0.35)
        ax5.plot(x, ate, 'D', color=clrs[j], ms=10, zorder=5)
        ax5.plot(x, lb, '_', color=clrs[j], ms=16, markeredgewidth=2)
        ax5.plot(x, ub, '_', color=clrs[j], ms=16, markeredgewidth=2)
        cov = res['Coverage']
        ax5.text(x, ub + 0.04, f'{cov*100:.0f}%', ha='center', fontsize=8,
                 color=clrs[j])
    ax5.set_xticks([1, 2, 3])
    ax5.set_xticklabels(['A\n(lower tight)', 'B\n(upper tight)', 'C\n(interior)'], fontsize=8)
    ax5.set_ylabel('Effect size', fontsize=9)
    ax5.set_title('Panel E\nSharp Bounds [ITT, ATT]\n(Theorem 2; diamond = true ATE)', fontsize=9)

    # ── Panel F: RMSE √n convergence (DGP1) ──────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(ns, df1['Overlay_RMSE'], 'o-', color=blue,   lw=1.8, ms=5,
             label='Overlay ATT̂ RMSE')
    ref = df1['Overlay_RMSE'].iloc[0] * np.sqrt(ns[0])
    ax6.plot(ns, ref / np.sqrt(ns), 'k--', lw=1, alpha=0.7,
             label='O(1/√n) reference')
    ax6.set_xscale('log'); ax6.set_yscale('log')
    ax6.set_xlabel('Sample size n', fontsize=9)
    ax6.set_ylabel('RMSE (log scale)', fontsize=9)
    ax6.set_title('Panel F\nOverlay ATT̂: √n Convergence\n(DGP1, Proposition 1)', fontsize=9)
    ax6.legend(fontsize=7)

    fig.suptitle(
        'Monte Carlo Simulations — "Measuring Causal Effects under Opaque Targeting"\n'
        'DGP1: Meta logistic  |  DGP2: TikTok ML  |  DGP3: Platform O (Ye et al. setting)',
        fontsize=11, fontweight='bold'
    )
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'mc_results.{ext}', bbox_inches='tight', dpi=200)
    print(f"Saved: {OUT}/mc_results.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 68)
    print("OVERLAY EXPERIMENT MONTE CARLO SIMULATIONS")
    print("Concrete Platform DGPs: Meta / TikTok / Platform O (Ye et al.)")
    print("=" * 68 + "\n")

    print("Computing true estimands via Monte Carlo integration (2M draws)...")
    ATT1        = true_att_dgp1()
    ATT2        = true_att_dgp2()
    ATT_A, ATT_B = true_att_dgp3()
    print(f"  DGP1  ATT = {ATT1:.6f}   ATE = {TRUE_ATE_DGP1:.6f}   "
          f"Selection ratio = {ATT1/TRUE_ATE_DGP1:.4f}")
    print(f"  DGP2  ATT = {ATT2:.6f}")
    print(f"  DGP3  ATT_A = {ATT_A:.6f}   ATT_B = {ATT_B:.6f}")
    print()

    df1 = run_exp1(ATT1, n_sims=500,
                   sample_sizes=[500, 2_000, 10_000, 50_000])

    df2 = run_exp2(ATT2, n_sims=400,
                   sample_sizes=[2_000, 10_000, 50_000])

    df3A, df3B = run_exp3(ATT_A, ATT_B, n_sims=400,
                           sample_sizes=[500, 2_000, 10_000, 50_000])

    df4, p_grid, mte_true, mte_mean, mte_sd = run_exp4(n=50_000, n_sims=200)

    exp5 = run_exp5(n=20_000, n_sims=500)

    # Save tables
    df1.to_csv(OUT / 'exp1_dgp1_meta_logistic.csv',   index=False)
    df2.to_csv(OUT / 'exp2_dgp2_tiktok_ml.csv',       index=False)
    df3A.to_csv(OUT / 'exp3_dgp3_firmA.csv',           index=False)
    df3B.to_csv(OUT / 'exp3_dgp3_firmB.csv',           index=False)
    df4.to_csv(OUT / 'exp4_mte_identification.csv',    index=False)

    make_figures(df1, ATT1, df2, ATT2,
                 df3A, df3B, ATT_A, ATT_B,
                 df4, p_grid, mte_true, mte_mean, mte_sd,
                 exp5)

    print("\nAll simulations complete.")


if __name__ == '__main__':
    main()
