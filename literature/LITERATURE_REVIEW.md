# Literature Review: Overlay Experiments and Identification under Opaque Targeting

This document surveys the papers most closely related to "Measuring Causal Effects under Opaque Targeting: Point and Partial Identification in Overlay Experiments." It is organized by stream and concludes with a positioning table.

---

## 1. Marginal Treatment Effects and the Local-IV Framework

### Heckman, J.J. and Vytlacil, E.J. (2005). "Structural Equations, Treatment Effects, and Econometric Policy Evaluation." *Econometrica* 73(3): 669–738.

The foundational paper for the MTE framework. Heckman and Vytlacil unify the LATE, IV, and selection model literatures by showing that all standard treatment effect parameters (ATE, ATT, ATU, LATE) can be written as weighted averages of the Marginal Treatment Effect MTE(x,u) = E[Y(1)-Y(0)|X=x, U=u], where U is the latent resistance to treatment. The MTE is identified as the derivative of E[Y|X=x, P=p] with respect to p, where P = P(D=1|X,Z) is the propensity score driven by an instrument Z. **Relation to our paper**: Section 4 of our paper directly applies this framework. Our "excluded shifter" S plays the role of the instrument Z, and the fact that W_i = D_i when Z_i = 1 means the econometrician directly observes the platform propensity score in the treated arm — a feature that simplifies the standard local-IV setup.

### Vytlacil, E.J. (2002). "Independence, Monotonicity, and Latent Index Models: An Equivalence Result." *Econometrica* 70(1): 331–341.

Shows that Imbens and Angrist's (1994) LATE assumptions (independence and monotonicity) are equivalent to a latent index model with a continuous instrument. This establishes that the latent index structure in our Assumption 3 (Excluded Shifter) is not an additional restriction beyond the standard IV conditions but rather an equivalent representation. **Relation to our paper**: Our Assumption 3 explicitly invokes the latent index model, making this equivalence directly relevant.

### Carneiro, P., Heckman, J.J., and Vytlacil, E.J. (2011). "Estimating Marginal Returns to Education." *American Economic Review* 101(6): 2754–2781.

Provides an empirical application of the MTE framework to returns to education, demonstrating how to nonparametrically estimate the MTE schedule and recover a range of treatment effect parameters. Illustrates that MTE estimates differ substantially across the range of U, confirming that selection on gains is economically relevant. **Relation to our paper**: Motivates our Assumption 4 (Monotone Selection on Gains) — the finding that individuals with stronger expected gains are more likely to be treated is robust across a range of economic applications.

### Angrist, J.D. and Imbens, G.W. (1994). "Identification and Estimation of Local Average Treatment Effects." *Econometrica* 62(2): 467–475.

Establishes the LATE (Local Average Treatment Effect) as the parameter identified by a binary IV in a setting with one-sided or two-sided noncompliance. Under monotonicity, the IV estimator identifies the effect for compliers. **Relation to our paper**: Our overlay experiment has a structure akin to two-sided noncompliance (since W_i = D_i * Z_i rather than the standard D_i(Z_i) setup), but Proposition 1 shows that the passive overlay in fact identifies the ATT — a different and richer object than LATE — because the overlay acts as a pure gate rather than a shifter.

---

## 2. Partial Identification

### Manski, C.F. (1990). "Nonparametric Bounds on Treatment Effects." *American Economic Review Papers and Proceedings* 80(2): 319–323.

The seminal paper on partial identification of treatment effects. Manski derives nonparametric bounds on ATE under no assumptions beyond the marginal distributions of observed outcomes. The bounds are [E[Y(1)|D=1]·P(D=1) - 1·P(D=0), E[Y(1)|D=1]·P(D=1) + 1·P(D=0)], with width equal to the range of outcomes. **Relation to our paper**: Our partial identification strategy differs by starting from the ITT (identified from the overlay) rather than observed outcome means, and by exploiting shape restrictions (monotone selection, non-negativity) rather than only outcome range restrictions. Our bounds [ITT(x), ATT(x)] are tighter than Manski's because the overlay design already identifies ATT.

### Manski, C.F. and Pepper, J.V. (2000). "Monotone Instrumental Variables: With an Application to the Returns to Schooling." *Econometrica* 68(4): 997–1010.

Introduces the monotone IV assumption as a shape restriction that tightens partial identification bounds. The intuition is that individuals who are more "resistant" to treatment tend to have lower (or higher) treatment effects. **Relation to our paper**: Our Assumption 4 (Monotone Selection on Gains) is a related shape restriction in the MTE framework — the platform assigns treatment to users with higher expected gains first (lower U values have higher MTE). This is precisely the economic rationale for an advertising platform optimizing ROI.

### Horowitz, J.L. and Manski, C.F. (2000). "Nonparametric Analysis of Randomized Experiments with Missing Covariate and Outcome Data." *Journal of the American Statistical Association* 95(449): 77–84.

Extends partial identification methods to settings with missing data, demonstrating that bounds remain informative even when key variables are unobserved. **Relation to our paper**: In the passive overlay, p(x) = P(D=1|X=x) is identified from the treated arm but the platform's full assignment rule is unobserved. Our bounds use only the observable ITT(x) and p(x), analogous to bounding with partially observed treatment.

---

## 3. Overlapping and Simultaneous Experiments on Platforms

### Tang, D., Agarwal, A., O'Brien, D., and Meyer, M. (2010). "Overlapping Experiment Infrastructure: More, Better, Faster Experimentation." *Proceedings of the 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 17–26.

Describes Google's engineering solution to the problem of running thousands of simultaneous A/B tests. Introduces "layers" and "domains" that partition traffic orthogonally across experiments, ensuring that different experiments are assigned independently. The key insight is that orthogonal assignment guarantees that each experiment's ITT estimator is unconfounded by other concurrent treatments. **Relation to our paper**: Tang et al. solve the DESIGN problem of running many experiments simultaneously (how to ensure independence). Our paper addresses the IDENTIFICATION problem that arises when a single experimenter overlays their design on a platform whose assignment rule is opaque — i.e., when the experimenter cannot enforce the orthogonal layering that Tang et al. describe.

### Xiong, T., Wang, Y., and Zheng, S. (2020). "Orthogonal Traffic Assignment in Online Overlapping A/B Tests." *EasyChair Technical Report*, Tencent Inc.

Describes Tencent's implementation of orthogonal traffic assignment for overlapping A/B tests, essentially the same engineering solution as Tang et al. applied at a platform running thousands of experiments. Confirms that this infrastructure is now standard practice at large tech companies. **Relation to our paper**: Further motivates our setting. While internal platform teams can enforce orthogonality, external advertisers (who are not part of the platform's experiment infrastructure) face exactly the opaque targeting problem our paper analyzes.

### Ye, Z., Zhang, Z., Zhang, D.J., Zhang, H., and Zhang, R. (2025). "Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence." *Management Science*, Articles in Advance.

**[Most closely related paper — detailed comparison below.]**

Ye et al. study a platform that runs m simultaneous A/B tests with orthogonal, independently randomized assignment. The treatment combination is T ∈ {0,1}^m, the assignment mechanism v(t|x) is **known**, and 2^m − (m+2) treatment combinations are **unobserved**. The DeDL framework uses DML + a structured DNN (generalized sigmoid link function) to identify and infer ATEs for all 2^m combinations from the m+2 observed ones. The approach yields √n-consistent, asymptotically normal estimators via influence functions (Neyman orthogonality). Validated empirically on a TikTok-like platform with 3 treatments (2^3=8 observable combinations as ground truth), DeDL substantially outperforms linear additivity (LA), linear regression (LR), and pure deep learning benchmarks.

**Detailed comparison with our paper:**

| Dimension | Ye et al. (2025) | This paper |
|---|---|---|
| **Who controls treatment** | Platform controls all m experiments | External experimenter overlays Z on opaque platform D |
| **Assignment mechanism** | Known: v(t\|x) known, orthogonally randomized | Unknown: p(x) = P(D=1\|X=x) is latent |
| **Core challenge** | 2^m combinations unobserved (completeness) | Opaque targeting induces endogenous selection (endogeneity) |
| **Method** | DML + DNN (machine learning) | IV/MTE theory; partial identification bounds |
| **Identification** | Point ID via interpolation across observed combinations | Point ID via local-IV (with excluded shifter); bounds (without) |
| **Key assumption** | v(t\|x) known; sufficient combinations observed | Latent index model; monotone selection on gains |
| **Object identified** | ATE(t) for all t ∈ {0,1}^m | MTE(x,u), ATT(x), ATE(x), bounds on ATE(x) |
| **Setting** | Inside the platform | Outside the platform (advertiser, auditor, researcher) |

**Complementarity**: The two papers address different layers of the same practical ecosystem. Ye et al. is the right tool when the *platform* wants to identify optimal treatment combinations from its own experiments. Our paper is the right tool when an *external experimenter* (advertiser, third-party auditor, smaller firm) needs to measure the causal effect of the platform's own targeting on their specific outcome. Crucially, Ye et al.'s framework requires knowing v(t|x) — precisely the information that is unavailable to an external experimenter facing an opaque platform.

**The multi-experiment story as bridge**: When Firm A and Firm B simultaneously overlay their experiments on the same platform, the platform's realized treatment vector for each firm depends on the other's experiment, the platform's own optimization, and both firms' targeting budgets. From Firm A's perspective, the effective D_i is shaped by a complex combination of all these forces — it is latent and opaque. This is the setting our paper addresses. Ye et al.'s setting is the special case where only one firm (the platform itself) controls all experiments and all assignment rules are known.

### Johari, R., Li, H., Liskovich, I., and Weintraub, G.Y. (2022). "Experimental Design in Two-Sided Platforms: An Analysis of Bias." *Management Science* 68(10): 7069–7089.

Studies bias in A/B tests on two-sided platforms (e.g., labor markets, ride-sharing) due to market interference: treating some users on one side of the market shifts outcomes for untreated users on the other side. Proposes bias-corrected estimators. **Relation to our paper**: Johari et al.'s interference bias and our selection bias are distinct problems arising from different platform features. Their interference is a SUTVA violation; ours is a one-sided noncompliance problem due to opaque targeting. However, both papers highlight that platform-mediated treatments introduce econometric complications that go beyond standard randomized experiments.

---

## 4. Platform Advertising and Field Experiments

### Gordon, B.R., Moakler, R., and Zettelmeyer, F. (2023). "Close Enough? A Large-Scale Exploration of Non-Experimental Approaches to Advertising Measurement." *Marketing Science* 42(4): 768–793.

A large-scale empirical comparison of observational and experimental estimates of advertising effectiveness. Documents that commonly used non-experimental methods (e.g., DML with observational data) substantially underestimate or overestimate ad effects compared to randomized experiment benchmarks, with errors often exceeding 100%. **Relation to our paper**: Motivates why the overlay design — which provides experimental variation Z while keeping the platform's targeting active — is more credible than observational approaches. It also motivates the gap between ATT (what the platform's targeting achieves for targeted users) and ATE (what a population-level campaign would achieve).

### Kohavi, R., Tang, D., and Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.

The definitive industry guide to running A/B tests at large tech platforms. Documents the organizational and statistical challenges of running hundreds to thousands of simultaneous experiments, including traffic allocation, variance reduction, and the orthogonal experiment infrastructure described in Tang et al. (2010). **Relation to our paper**: Provides institutional background supporting our motivation. The book's discussion of overlapping experiments — where different teams and advertisers run experiments simultaneously — directly motivates why the overlay design studied in our paper arises in practice.

---

## 5. Double Machine Learning and Semiparametric Estimation

### Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). "Double/Debiased Machine Learning for Treatment and Structural Parameters." *Econometrics Journal* 21(1): C1–C68.

The foundational DML paper. Shows that Neyman orthogonality (influence functions) combined with cross-fitting eliminates the first-order bias from ML nuisance parameter estimation, yielding √n-consistent, asymptotically normal estimators of treatment effects even when nuisance functions are estimated at slower ML rates. **Relation to our paper**: The DML framework is used by Ye et al. (2025) for the combinatorial experiment setting. In our paper, the MTE (Section 4) is identified as a derivative of a conditional expectation — no first-stage ML estimation is required because p(x) = P(W=1|Z=1,X=x) is directly identified. Our paper thus provides the identification foundation on which a DML-style estimator could be built for the overlay setting.

---

## Summary Positioning Table

| Stream | Key papers | What they do | Our paper's contribution |
|---|---|---|---|
| MTE / Local-IV | Heckman-Vytlacil (2005), Vytlacil (2002), Angrist-Imbens (1994) | Identification via continuous instrument | Applies MTE to overlay design; excluded shifter = instrument; overlay reveals propensity directly |
| Partial ID | Manski (1990), Manski-Pepper (2000) | Bounds under no/weak assumptions | Sharp bounds [ITT, ATT] from overlay + monotone selection + nonnegativity; tighter than Manski using overlay structure |
| Platform overlapping experiments | Tang et al. (2010), Xiong et al. (2020), Kohavi et al. (2020) | Engineering solution for simultaneous experiments | Analyzes what happens when experimenter CANNOT enforce orthogonality; opaque platform targeting |
| Combinatorial experiments | Ye et al. (2025) | DML+DNN for multiple simultaneous A/B tests with known assignment | Addresses case where assignment is UNKNOWN (opaque); provides identification theory where Ye et al. requires known v(t\|x) |
| Platform interference | Johari et al. (2022) | Bias from market-level interference in two-sided platforms | Different problem (SUTVA vs. noncompliance) but related platform-mediated treatment challenge |
| Ad measurement | Gordon et al. (2023) | Shows observational methods fail for ad effects | Motivates why overlay experiment is needed; explains ATE vs ATT gap in advertising context |
