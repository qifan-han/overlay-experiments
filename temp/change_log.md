# Change Log

## 2026-04-09

### Major revision of `paper/main.tex`
Complete structural rewrite following journal norms learned from Ye et al. (2025, MS) and Waisman & Gordon (2025, MS).

**Structural changes:**
- New section order: Introduction → Literature Review → Theory → Monte Carlo Simulation → Conclusion
- Proofs moved from main body to Appendix
- Added Monte Carlo Simulation section (Section 4) with full DGP specification and justification
- Added Conclusion with dedicated Managerial Implications and Limitations subsections

**Terminology fixes:**
- `ATE(x)` → `CATE(x)` (Conditional Average Treatment Effect) throughout
- `ATT(x)` → `CATT(x)` (Conditional Average Treatment Effect on the Treated) throughout
- `ITT(x)` → `CITT(x)` (Conditional Intent-to-Treat effect) throughout
- `p(x)` → `e(x)` for the passive-overlay propensity (following Wager 2024 convention)
- `p(x)` freed; `π(x,s)` retained for the MTE propensity in Section 3.3

**Introduction changes:**
- Contributions described in narrative prose only — no `\ref{}` cross-references to theorems
- Added managerial framing: CATT vs. CATE for scale-up decisions
- Added explicit contrast with Waisman & Gordon (2025) and Ye et al. (2025)

**Literature Review changes:**
- Zero equations in literature review (all conceptual/prose)
- H-V instrument notation clarified to avoid collision with paper's Z variable
- Added Waisman & Gordon (2025) as the closest related paper with detailed positioning

**Theory section changes:**
- Added substantive discussion after each Assumption: why it is reasonable, when it may fail
- Added discussion after each Proposition/Theorem: economic interpretation, managerial content
- Added practical examples of excluded shifters (geographic auction competition, temporal platform load, cross-market budgets)
- Clarified operational advantage of overlay design vs. standard IV (propensity directly observed in eligible arm)
- Remark added clarifying which assumptions are needed for each bound

**Monte Carlo section (new):**
- 3 DGPs with full parameter values and economic justification
- DGP 1: Baseline, monotone nonneg MTE, calibrated to Gordon et al. (2019) effect sizes
- DGP 2: Calibrated to Facebook data following Waisman & Gordon, non-monotone
- DGP 3: Nonneg effects violated — stress test
- Excluded-shifter geographic design specified
- Table/figure placeholders for results

**Conclusion changes:**
- Three managerial recommendations with explicit decision-problem framing
- Five limitations identified with future research directions

### Updated `paper/refs.bib`
- Added `waisman2025multicell`: Waisman & Gordon (2025, MS)

### Added natbib to preamble
- Replaced manual `Author (Year)` citations with `\citet{}` / `\citep{}` commands
- Added `\bibliographystyle{apalike}` and `\bibliography{refs}`
