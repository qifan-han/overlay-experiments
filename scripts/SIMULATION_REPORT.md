# Monte Carlo Simulation Report
## "Measuring Causal Effects under Opaque Targeting"

**Date:** April 2026  
**Script:** `scripts/simulation.py`  
**Figures:** `scripts/figures/mc_results.pdf`

---

## Design Philosophy

Each simulation uses a **concrete platform targeting algorithm as the DGP** — the
same opaque rule that the econometrician cannot observe. Estimators are applied
without knowledge of the DGP; results are evaluated against true estimands computed
by 2-million-draw Monte Carlo integration. Three naive benchmarks are compared
against the correct overlay estimator to demonstrate what goes wrong without the
overlay design.

This approach directly mirrors Ye et al. (2025), who simulate from their
generalized sigmoid DGP and evaluate DeDL against benchmarks. The key difference:
Ye et al. assume v(t|x) is **known** to the platform; our simulation demonstrates
that the overlay estimator works correctly when v(t|x) (or equivalently p(x)) is
**unknown** to the external experimenter.

---

## Three Data-Generating Processes

### DGP 1 — Meta/Facebook Advertiser: Logistic Targeting

**Real-world context:** A DTC brand runs a conversion lift test on Meta. Meta's ad
auction scores users on purchase intent X ~ U[0,1] and targets them via a logistic
rule. Two forms of selection operate simultaneously: (1) high-intent users have
higher baseline conversions (selection into treatment on observables), and (2)
high-intent users also respond more strongly to ads (selection on gains). The brand
observes (Y, Z, W=D·Z) but never sees Meta's propensity function p(X) or individual
targeting status D when Z=0.

**Specification:**
- X ~ Uniform[0,1] (purchase intent score)
- p(X) = sigmoid(−2 + 4X): low-intent users p ≈ 12%, high-intent p ≈ 88%
- τ(X) = 1 + 2X (selection on gains: high-intent users respond more)
- Y₀ = 0.5·X + ε, ε ~ N(0,1) (baseline correlated with X — selection in Y₀)
- D = 1{U < p(X)}, Z ~ Bernoulli(0.5), W = D·Z

**True estimands (numerical):** ATT = 2.2815, ATE = 2.0000, Selection ratio = 1.141

---

### DGP 2 — TikTok External Advertiser: Nonlinear ML Targeting

**Real-world context:** A music label tests a promotional campaign on TikTok. TikTok's
ranking engine determines targeting via a nonlinear interaction of user age (X₁) and
content affinity (X₂). Young users with high affinity get disproportionately high
targeting probability; the negative interaction term suppresses it for older
high-affinity users — a pattern common in neural-network ranking models. The label
cannot invert or approximate TikTok's model.

**Specification:**
- X₁, X₂ ~ Uniform[0,1] independently
- p(X₁,X₂) = sigmoid(−1 + 3X₁² + 2X₂ − 4X₁X₂) (nonlinear, non-separable)
- τ(X₁,X₂) = 1 + X₁ + X₂ (heterogeneous treatment effect)
- Y₀ = 0.3X₁ + 0.3X₂ + ε

**True estimands (numerical):** ATT = 2.0400

---

### DGP 3 — Platform O: External Advertisers Alongside Ye et al.'s m=3 Setup

**Real-world context:** Platform O (TikTok-like, as studied by Ye et al. 2025) runs
m=3 internal A/B tests whose outcome follows Ye et al.'s generalized sigmoid form II
(their equation 6). Two external advertisers — Firm A (electronics) and Firm B
(fashion) — simultaneously run overlay experiments. The platform allocates ad slots
by auction: Firm A wins when its score (driven by purchase intent X₁) exceeds Firm
B's (driven by brand affinity X₂). This means Firm A's targeting probability p_A
depends on X₂, which Firm A cannot observe.

From Firm A's perspective, the entire system — Platform O's three internal
experiments, Firm B's targeting, and their interaction — is a black box. Ye et al.'s
DeDL framework would require knowing v(t|x) to identify effects for all 2^3=8
combinations. Firm A needs only ATT_A and can obtain it with their overlay.

**Specification:**
- Firm A's targeting: p_A(X₁,X₂) = sigmoid(2X₁ − 1.5X₂)
  (positive purchase intent, negative competition from Firm B)
- Firm B's targeting: p_B(X₁,X₂) = sigmoid(2X₂ − X₁ + 0.3)
- Platform O's other experiments: T₂,T₃ ~ Bernoulli(0.6) independently
- Outcome model: Ye's generalized sigmoid form II, θ₄=5, θ₀(X)=−1+0.8X₁+0.5X₂,
  θ₁(X)=1+1.5X₁, θ₂(X)=0.8+0.5X₂, θ₃=0.6
- Each firm's own outcome (purchases/brand recall) depends only on their W

**True estimands (numerical):** ATT_A = 1.8522, ATT_B = 1.3265

---

## Results

### Experiment 1: Proposition 1 — ATT Identification (DGP 1)

| n | Overlay ATT (correct) | Bias | OLS (biased ↑) | Bias | Plain ITT (biased ↓) | Bias |
|---|---|---|---|---|---|---|
| 500 | 2.2904 | +0.0089 | 2.3739 | +0.0925 | 1.1471 | −1.1344 |
| 2,000 | 2.2821 | +0.0006 | 2.3766 | +0.0951 | 1.1404 | −1.1411 |
| 10,000 | 2.2804 | −0.0011 | 2.3745 | +0.0930 | 1.1408 | −1.1406 |
| 50,000 | 2.2820 | +0.0005 | 2.3758 | +0.0944 | 1.1407 | −1.1407 |

**True ATT = 2.2815**

**Assessment:** ✅ **Consistent with Proposition 1.**

Three distinct behaviors confirm the theory:

1. **Overlay ATT estimator**: Bias collapses to zero as n grows (< 0.001 at n ≥ 2,000); RMSE shrinks at the √n rate (Panel F). This is the correct estimator.

2. **Naive OLS**: Persistent bias of +0.093 regardless of n. The bias does not shrink because it stems from an omitted variable (X affects both D and Y₀), not from sampling error. OLS *consistently* estimates the wrong quantity — it overestimates ATT by attributing the baseline advantage of high-intent users to the ad campaign.

3. **Plain ITT**: Persistent bias of −1.14 regardless of n. ITT consistently estimates p̄ × ATT ≈ 0.5 × 2.28 = 1.14, not ATT. Dividing by p̂ corrects this exactly.

The selection ratio ATT/ATE = 1.141 confirms that Meta's targeting achieves 14.1% better lift than a uniform campaign would — the economic value of algorithmic selection on gains.

---

### Experiment 2: Proposition 1 Robustness — Nonlinear ML Targeting (DGP 2)

| n | Overlay ATT | Bias | OLS | Bias | Plain ITT | Bias |
|---|---|---|---|---|---|---|
| 2,000 | 2.0331 | −0.0069 | 2.0534 | +0.0134 | 1.0120 | −1.0280 |
| 10,000 | 2.0415 | +0.0016 | 2.0567 | +0.0167 | 1.0172 | −1.0227 |
| 50,000 | 2.0412 | +0.0012 | 2.0568 | +0.0169 | 1.0167 | −1.0233 |

**True ATT = 2.0400**

**Assessment:** ✅ **Consistent with Proposition 1 (robustness confirmed).**

The overlay estimator is correct despite the nonlinear, non-separable, two-dimensional targeting rule. Neither the functional form of p(·,·) nor the dimensionality of X matters — because Proposition 1 is a model-free identity: ITT/p̄ = ATT regardless of how p was generated. OLS and plain ITT remain persistently biased, confirming the failure mode is structural, not finite-sample.

---

### Experiment 3: Proposition 1 in Multi-Firm Setting (DGP 3 — Platform O)

| n | ATT_A estimate | Bias | ATT_B estimate | Bias |
|---|---|---|---|---|
| 500 | 1.8524 | +0.0003 | 1.3297 | +0.0032 |
| 2,000 | 1.8613 | +0.0091 | 1.3234 | −0.0030 |
| 10,000 | 1.8517 | −0.0005 | 1.3295 | +0.0030 |
| 50,000 | 1.8503 | −0.0018 | 1.3266 | +0.0002 |

**True ATT_A = 1.8522; True ATT_B = 1.3265**

**Assessment:** ✅ **Consistent with Proposition 1 in the multi-firm setting.**

Both firms correctly and independently identify their own ATT. This holds even though:
- Firm A's targeting probability p_A(X₁,X₂) depends on X₂ (Firm B's covariate), which Firm A cannot observe
- Platform O runs three concurrent A/B tests (Ye et al.'s exact setup) with an opaque outcome model
- The two firms' targeting decisions are negatively correlated through auction competition

Each firm's overlay is independent (Z_A ⊥ Z_B), so their estimators are uncorrelated. The contrast with Ye et al.'s DeDL is sharp: DeDL would require Firm A to know v(t|x) for all eight treatment combinations — precisely what is unavailable to an external advertiser.

---

### Experiment 4: Theorem 1 — MTE Point Identification

**DGP:** Budget levels S ∈ {0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85} as geographic-market instruments; P = S; τ(U) = 2(1−U); **True MTE(p) = 2(1−p)**

| p | True MTE | Estimated | Bias | SD | RMSE |
|---|---|---|---|---|---|
| 0.20 | 1.6000 | 1.5299 | −0.0701 | 0.1790 | 0.1923 |
| 0.30 | 1.4000 | 1.3921 | −0.0079 | 0.1451 | 0.1453 |
| 0.40 | 1.2000 | 1.2000 | −0.0000 | 0.1260 | 0.1260 |
| 0.50 | 1.0000 | 1.0007 | +0.0007 | 0.0916 | 0.0916 |
| 0.60 | 0.8000 | 0.7928 | −0.0072 | 0.1237 | 0.1239 |
| 0.70 | 0.6000 | 0.5824 | −0.0176 | 0.1211 | 0.1224 |
| 0.80 | 0.4000 | 0.4594 | +0.0594 | 0.1636 | 0.1740 |

**Assessment:** ✅ **Consistent with Theorem 1.**

The derivative estimator recovers the MTE schedule across the interior of the propensity support. Bias at interior points (p ∈ [0.3, 0.7]) is negligible relative to RMSE. The boundary bias at p = 0.2 and p = 0.8 reflects the well-known finite-sample limitation of kernel regression at support endpoints (fewer observations contribute to local estimates). The practical recommendation remains: for MTE estimation, use sample sizes of n ≥ 20,000 and data-driven bandwidth selection in empirical applications.

---

### Experiment 5: Theorem 2 — Sharp Bounds [ITT, ATT]

**Fixed p̄ = 0.4; True: ITT = 0.6400, ATT = 1.6000**  
(n = 20,000 per replication, 500 replications)

| DGP | True ATE | LB mean | UB mean | Coverage |
|---|---|---|---|---|
| A: only targeted respond (ATE = ITT) | 0.6400 | 0.6405 | 1.6020 | **49.6%** |
| B: uniform response (ATE = ATT) | 1.6000 | 0.6392 | 1.5987 | **47.4%** |
| C: decreasing MTE (ATE interior) | 1.0000 | 0.6404 | 1.6007 | **100.0%** |

**Assessment:** ✅ **Consistent with Theorem 2, and confirms sharpness.**

- **DGP-C (interior):** 100% coverage. The sample bound [LB̂, ÛB] always contains the true ATE when it lies strictly inside [ITT, ATT]. ✓

- **DGP-A (lower endpoint tight):** Coverage ≈ 50%. ATE = ITT = 0.64 exactly. Due to sampling error, LB̂ = ITT̂ oscillates around 0.64; in ~50% of samples LB̂ > 0.64, pushing the bound above the true ATE. This is **sharpness**: the lower bound cannot be improved, because there exists a DGP (this one) consistent with the data that places ATE exactly at ITT. ✓

- **DGP-B (upper endpoint tight):** Coverage ≈ 47%. Symmetric logic: ATE = ATT = 1.6; in ~50% of samples ÛB < 1.6. Again confirms sharpness of the upper bound. ✓

The ~50% coverage at the endpoints is **not a failure** — it is the precise finite-sample signature of sharp bounds. A population-level identified set cannot be consistently estimated to contain a boundary point at higher than 50% coverage.

---

## Overall Conclusions

| Result | DGP | Theoretical Claim | MC Verdict |
|---|---|---|---|
| Proposition 1 (simple) | DGP1: Meta logistic | ITT/p →_p ATT; OLS and ITT fail | ✅ Confirmed |
| Proposition 1 (robustness) | DGP2: TikTok ML | Holds for nonlinear opaque rules | ✅ Confirmed |
| Proposition 1 (multi-firm) | DGP3: Platform O | Holds under auction competition | ✅ Confirmed |
| Theorem 1 | Budget-level shifter | ∂/∂p μ₁(p) identifies MTE(p) | ✅ Confirmed (boundary bias noted) |
| Theorem 2 (interior) | DGP-C | ATE ∈ [ITT, ATT], 100% coverage | ✅ Confirmed |
| Theorem 2 (sharpness lower) | DGP-A | ATE = ITT achievable; ~50% coverage | ✅ Confirmed |
| Theorem 2 (sharpness upper) | DGP-B | ATE = ATT achievable; ~50% coverage | ✅ Confirmed |

**Key simulation-level insight vs. Ye et al. (2025):** Ye et al.'s DeDL correctly
identifies all 2^m ATEs when v(t|x) is known. Our simulations demonstrate that when
v(t|x) is unknown — the external advertiser's situation — a simple overlay estimator
identifies the ATT without any knowledge of the platform's assignment mechanism.
DGP 3 directly instantiates this comparison: Platform O runs Ye et al.'s exact
generalized sigmoid DGP internally, but each external firm needs only their overlay
to identify their own ATT. The two approaches are complementary, not competing.
