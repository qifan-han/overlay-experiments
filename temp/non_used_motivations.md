# Non-Used Motivations, Examples, and Extensions

This file records ideas discussed during paper development that were ultimately
dropped from the current draft, along with the reasons for exclusion and notes
on what would need to be done to pursue them in other work.

---

## 1. OpenAI Subscription Payment Example

**What it was**: A researcher recruits participants online and randomly pays
(or blocks) OpenAI subscriptions. The researcher observes the platform's
targeting decision D_i (OpenAI runs ads for non-subscribers) but not the
targeting rule. Structurally, W_i = D_i * Z_i where Z_i = subscription status
the researcher controls.

**Why it was considered**: Strong adversarial framing that makes opacity natural.
The policy question (causal effect of ads on non-subscribers) is crisp, and the
structural setup maps exactly to the paper's framework.

**Why it was dropped**: The exclusion restriction is violated. The researcher's
subscription payment directly affects Y_i through non-ad channels — e.g., paying
for a subscription changes the user's product experience, content access, and
behavior in ways unrelated to ad exposure. Z_i affects potential outcomes
Y_i(0) and Y_i(1) directly, not only through W_i. No fix is available without
choosing an outcome variable entirely unaffected by subscription status, which
would likely be too narrow to be interesting.

**What would be needed to use it**: An outcome that is plausibly unaffected by
subscription status itself (e.g., purchases of a third-party product advertised
on the platform). This would need careful justification.

**Better alternative used instead**: Browser extension randomization (Section 1
of the paper): a researcher randomizes ad exposure by randomly enabling or
blocking ads via a browser extension for consenting participants. This is a clean
design with no exclusion restriction violation, and it preserves the third-party
measurement motivation.

---

## 2. LLM-Based Propensity Imputation

**What it was**: Use a large language model to impute the platform's targeting
propensity e(x) for users not in the eligible arm, by querying the LLM with user
feature descriptions and asking it to predict ad delivery probability. A
simulation was proposed with a calibration quality parameter ρ measuring the
correlation between LLM-imputed and true propensity.

**Why it was considered**: Novel, technically interesting, and potentially
democratizing for small advertisers and academic researchers without platform
API access.

**Why it was dropped**: Access paradox. Advertisers who have API access
(e.g., Meta Audience Insights, Google Reach Planner) do not need LLM imputation
— they can directly query the platform for reach and propensity estimates.
Researchers who lack API access cannot calibrate the LLM outputs, because
calibration requires knowing the true propensity for at least some users. The
approach serves no group that cannot be served better by an alternative. One
sentence in Limitations would suffice; including it as a contribution invites
immediate critique.

**What would be needed to use it**: A setting where (a) the researcher cannot
access platform APIs but (b) can validate LLM outputs against a ground truth for
a subset of users. This seems contrived.

**Potential future direction**: If LLMs gain direct access to platform APIs
(e.g., an "advertising oracle" model), this becomes more viable. A simulation
DGP parameterizing ρ (calibration quality) and finding the ρ* threshold below
which LLM imputation is no better than passive bounds would be a clean result.

---

## 3. Competing Firms Running Simultaneous Experiments (SUTVA Violation via Auction)

**What it was**: Two firms (say, Apple and Samsung) simultaneously run overlay
experiments targeting the same users on the same platform. The firms' targeting
decisions interact through the ad auction: if Apple's campaign increases bids
in a market, it raises Samsung's auction cost and reduces Samsung's delivery
probability, violating SUTVA for Samsung's experiment.

**Why it was considered**: Real and well-documented problem in industry (many
firms cite headaches from overlapping experiments). Motivated by the
multi-experiment literature (Tang & Kohavi; Ye et al. 2025).

**Why it was dropped**: Adding this as motivation invites the reviewer question
"why don't you model the interference?" Answering that question requires a
structural auction model — a full separate paper. The current paper's SUTVA
assumption then appears to be evading the hardest case rather than addressing it.
Furthermore, the CATT-CATE gap problem and the competing-firms problem are
orthogonal; conflating them muddies both contributions.

**What is handled in the current paper**: Within-firm simultaneous experiments
(marketing team's coupon overlay on top of recommendation team's algorithm),
where the two randomizations are independent and SUTVA holds by design. This is
mentioned in the Introduction as one of the three motivating settings.

**Potential future direction**: Combine the opaque targeting framework with an
auction interference model (Johari et al. 2022). The key challenge is that the
treatment propensity e(x) would no longer be a fixed function of x but would
depend on the competing firm's budget allocation — a game-theoretic equilibrium
problem. One paragraph in the current paper's Limitations (Section 5.2) notes
this direction.

---

## 4. Two-Environment EU/US Recruitment Design

**What it was**: Recruit participants from (a) EU (where the platform does not
run the competing advertiser's ads) and (b) US (where it does). By comparing
the Wald ratio across environments, the analyst identifies the isolated direct
effect of the focal ad separately from the interaction with the competing
treatment.

**Why it was considered**: A potential design fix for the SUTVA-violation problem
when competing firms run simultaneous campaigns targeting the same users.

**Why it was dropped**: Fundamentally broken. If the platform does not run ads in
the EU, then D_i = 0 for all EU participants, e(x) = 0, and there is no
treatment variation. No Wald ratio can be computed. Even if patched (e.g., the
focal firm's own ads still run in EU), the US/EU comparison requires parallel
trends or environmental homogeneity — that is, the CATE distribution must be
identical across environments. This is an empirical assumption requiring
justification, and the resulting estimator reduces to a difference-in-differences
structure with no structural identification content beyond what DiD already
provides.

**What would be needed to use it**: Markets where the competing firm does not
operate but the focal firm does, with strong evidence that user response
distributions are comparable. This is a design question, not an econometric
contribution.

---

## 5. Pure Econometrician Case (Observing Platform's Own A/B Test)

**What it was**: An econometrician who knows that the platform is running an A/B
test uses the overlap of their own sample with the platform's A/B test to create
groups. The econometrician does not observe the platform's assignment T_i^P.

**Why it was considered**: Natural generalization of the paper's setting; the
question of what is identified when both parties are running experiments is
interesting.

**Why it was dropped**: This is a strictly harder and structurally different
problem. The platform's assignment T_i^P is unobserved for all units. The
analyst's randomization Z_i is independent of T_i^P by construction (since the
platform is running its own A/B test unrelated to the analyst). Z_i therefore
cannot serve as an instrument for T_i^P. No vanilla ATE/ATT/LATE of T_i^P is
identified without additional structure. This is a different paper.

**What distinguishes the current paper**: In the current paper, the platform's
targeting decision D_i operates within the analyst's campaign — D_i = 1 means
the platform delivered the analyst's own campaign to user i. So D_i is observed
through W_i = D_i when Z_i = 1, and the analyst's eligibility indicator Z_i is
a valid instrument for W_i. The paper's framework is specifically about the
analyst's own campaign targeting, not about observing an external experiment
conducted by the platform.

---

## 6. SUTVA Extensions and Marginal Estimands Under Interaction

**What it was**: Under orthogonal eligibility designs (Z^A ⊥ Z^B for two firms),
the Wald ratio identifies CATT^{A,marginal} = the CATT averaging over the
equilibrium distribution of the competing treatment. Sharp bounds
[CITT^A, CATT^{A,marginal}] still hold for this marginal estimand.

**Why it was considered**: Formally correct. A genuine extension of the
paper's framework to the case where multiple treatments interact.

**Why it was dropped**: The marginal estimand is not policy-relevant when the
competitive environment is time-specific. An advertiser wants the isolated effect
of their own ads, not the effect averaged over the particular competitive
configuration that happened to prevail during the experiment. The extension also
substantially complicates the exposition for a result that does not deliver the
quantity the advertiser actually wants. The SUTVA approximation (small experiment
in a large market → interactions negligible) is the correct practical answer and
requires only one sentence.

---

## 7. API-Based Propensity from Platform Audience Insights

**What it was**: Meta Audience Insights, Google Reach Planner, and analogous
tools expose estimated reach (and thus delivery propensity) as a function of
targeting parameters. Advertisers can query these APIs for different audience
configurations to impute e(x) more precisely than observing the passive overlay.

**Why it was considered**: Potentially valuable for operationalizing the paper's
framework without a formal excluded shifter.

**Why it was dropped**: This is an engineering detail, not an identification
result. If the platform's API reveals e(x) accurately, the analyst already has
the treatment rate needed for Track B without any new theory. If the API reveals
e(x) conditional on targeting parameters (i.e., π(x,s) for different budget
settings), the analyst is effectively in the W&G setting (engineered propensity
variation). Neither case contributes to the paper's identification theory.

**Where it belongs**: A practitioner's guide or implementation note. One sentence
in the Estimation section noting that e(x) can sometimes be cross-validated
against platform-reported reach estimates would suffice.
