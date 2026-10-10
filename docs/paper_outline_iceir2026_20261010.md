# Paper outline for ICEIR 2026 (direction A + C)

## Material Passport

Origin: ARS `academic-paper`, `outline-only` mode (Phases 0–2), Claude Code session, 2026-10-10.
Paper Configuration Record confirmed by the user on 2026-10-10 ("Xác nhận, làm dàn ý ngay").
Status: **outline approved by the user on 2026-10-10** ("Duyệt dàn ý, chọn tiêu đề 2, bắt đầu viết
bản thảo"); title 2 selected. No section had been drafted when this outline was approved. No experiment, test
access or code change is implied by this document.
Evidence base: the five closed result documents in `docs/` and the 2026-10-03 literature
review. Every number below is quoted from those documents; none is new.
`criteria_binding_unavailable`: no review-criteria registry entry was bound for this venue, so
this outline makes no venue-alignment claim beyond the stated submission rules.

## Paper Configuration Record (confirmed)

| Parameter | Value |
|---|---|
| Topic | Lightweight SAR urban-flood change detection: a controlled negative result, seed and event sensitivity, and an accuracy–cost benchmark |
| Research questions | RQ1: Do size weighting and a boundary package have a consistent effect across seeds, two training recipes and a held-out test set? RQ2: How stable are method rankings against seed and event variation? RQ3: What is the accuracy–cost trade-off of sub-5M-parameter change detectors on GPU and CPU? |
| Paper type | Conference paper, IMRaD structure |
| Discipline | Remote sensing / deep learning for SAR; read by a general AI audience |
| Target venue | ICEIR 2026 (Springer LNNS), submission deadline 2026-10-15 |
| Venue rules (as stated by the venue) | English; 12–15 pages; Springer LNNS template, LaTeX preferred; PDF via Microsoft CMT; original unpublished work. Review model not stated. |
| Citation format | Numbered, as in the LNNS template |
| Output format | Markdown outline; LaTeX (LNNS template) for the draft |
| Body language / abstract | English / English only |
| Word count target | about 6,000 words (planning estimate for 12–15 LNNS pages with tables; the binding limit is pages) |
| Co-authors, funding | To be supplied by the user |
| Citation verification, retraction policy | advisory (mark only, default) |
| Operational mode | outline-only |

### Notes: fixed claim limits

The paper must **not** claim: that size weighting or the boundary package improves any
metric; that recipe R1 (or any recipe) solves the instability; that BIT-SAR v2's robustness
is due to its architecture; anything about Jetson Orin, TensorRT or real-time operation;
that seed variation is caused by low flood fraction (test rule T3 did not generalize); any
p-value or confidence interval from pixels, patches or components; any test result beyond
the single pre-specified evaluation.

## Paper Outline

### Structure pattern: Conference paper (IMRaD hybrid)

### Working titles (title 2 chosen)

1. *When Small-Object Losses Do Not Help: A Controlled, Multi-Seed Study of Lightweight SAR
   Urban Flood Change Detection*
2. *Seed and Event Sensitivity in Lightweight SAR Flood Change Detection: A Controlled
   Negative Result and an Accuracy–Cost Benchmark*
3. *Parameters Are Not Cost, One Seed Is Not Evidence: Benchmarking Lightweight SAR Flood
   Change Detectors*

### Overview

The paper reports what happened when a lightweight, small-region-aware SAR flood change
detector was tested under pre-specified protocols rather than a single run. It introduces the
task and the two proposed components (size-weighted loss, boundary package), then describes a
staged design: a 2×2 factorial against two lightweight baselines, six seeds, two training
recipes, sign-based decision rules fixed before each stage, and one sealed test evaluation.
Results show (i) no consistent effect of either component on validation or test, (ii) rank
instability among the factorial arms but a stable order between model families, (iii) seed
variation concentrated in a single event on each split, without support for a flood-fraction
explanation, and (iv) a cost table in which parameter count, FLOPs, GPU latency and CPU
latency rank the models differently. The discussion states what is and is not established and
gives reporting recommendations.

### Review Criteria Coverage Plan

`criteria_binding_unavailable`. No criterion ids are bound; scientific validity, venue fit
and submission readiness are tracked only through the venue's stated rules (12–15 pages,
English, LNNS template).

### Detailed Outline

#### Abstract (150–200 words, not counted)
Problem (small urban floods in SAR; lightweight models); what was tested (two components in a
2×2 factorial, two baselines, six seeds, two recipes, one sealed test); main findings (no
consistent component effect; all models recall 2–7% of interior small floods on test; stable
family order BIT-SAR v2 > FC-Siam-Diff > SmallFlood-CDNet; single-event concentration of seed
variation; cost ranking depends on device); implication (multi-seed, per-event and multi-cost
reporting). Keywords: SAR, flood mapping, change detection, lightweight models, seed
sensitivity, negative result, efficiency.

#### 1. Introduction (~800 words)
**Purpose**: motivate the study for a general AI audience and state the contributions.
**Serves sub-question**: framing (RQ1–RQ3 introduced).
**Content**:
- 1.1 Why SAR for floods, and why urban and small floods are the hard case
  - SAR sees through cloud; urban flooding is the weak point of current benchmarks.
- 1.2 Why lightweight models, and why "lightweight" needs more than a parameter count
- 1.3 The initial hypothesis and what changed
  - Two plausible components (size-aware Tversky weighting; thin boundary head and loss).
  - A single-seed pilot looked interpretable; confirmation runs did not agree.
- 1.4 Contributions (stated as findings, not as a new method)
  - C1: a controlled negative result for both components across 6 seeds, 2 recipes and a
    sealed test set.
  - C2: an analysis of seed and event sensitivity, including a pre-specified explanation
    that the test set did not support.
  - C3: an accuracy–cost benchmark of three model families with a full cost set.
  - C4: released protocols, decision rules and code for reproduction.
**Sources**: Zhao et al. 2024; Bonafilia et al. 2020; Li et al. 2019; Dehghani et al. 2022;
Corley et al. 2024; seed-variance literature (to verify, see Evidence Map).
**Transition to next**: the contributions depend on what prior work did and did not test.

#### 2. Related Work (~750 words)
**Purpose**: position the study; support each "not previously tested" statement with a
documented search bound.
**Serves sub-question**: framing for RQ1–RQ3.
**Content**:
- 2.1 SAR flood datasets and urban flooding (Sen1Floods11, UrbanSARFloods, Kuro Siwo,
  S1GFloods)
- 2.2 Bitemporal change detection and simple baselines (FC-Siam-Diff, BIT; "reality check"
  results that plain models remain competitive)
- 2.3 Small-object and boundary-aware losses (Tversky, blob loss, instance-wise losses,
  boundary loss), mostly from medical imaging and optical change detection
- 2.4 Efficiency reporting (the efficiency misnomer; lightweight change detectors; onboard
  flood mapping)
- 2.5 Variance across seeds and reporting practice in machine learning
- Gap statement, bounded: "within the searches reported in our review (31 queries, one
  engine, 2026-10-03), we found no SAR flood study that reports interior-small recall or
  multi-seed rank stability."
**Sources**: all Theme 1–7 sources of the literature review; seed-variance sources.
**Transition to next**: the gap motivates a design that fixes decisions in advance.

#### 3. Experimental Design (~1,300 words)
**Purpose**: give enough detail to reproduce and to judge the claims.
**Serves sub-question**: RQ1–RQ3 (methods).
**Content**:
- 3.1 Data and splits
  - UrbanSARFloods Sentinel-1 intensity, pre/post VH+VV, 256×256 patches.
  - Event-held-out split: 12 train events (19,166 patches), 3 validation (4,767), 3 test
    (9,204). State that this repartition differs from the dataset's official split, so
    numbers are not comparable with the dataset paper.
- 3.2 Models (Table 2)
  - SmallFlood-CDNet (MobileNetV3-small Siamese encoder, 1.08M parameters) in a 2×2
    factorial: A baseline, B +size weighting, C +boundary package, D both.
  - FC-Siam-Diff (0.49M) and BIT-SAR v2 (3.49M, adapted from BIT).
  - Whole-method caveat: baselines differ in loss coefficients, initialization and runner.
- 3.3 Training and determinism
  - 15 epochs, batch 8, AdamW, FP32, strict determinism; paired initialization and batch
    order across A–D within a seed.
  - Two recipes: original (3e-4 for A–D and FC; 1e-4 for BIT-SAR v2) and R1 (1e-4 for all),
    R1 selected on an internal dev split of training events only.
- 3.4 Metrics
  - Event-macro pixel F1 (primary), per-event F1, component F1, interior-small and
    clipped-small recall, boundary F1, inner-band IoU; threshold 0.5; last checkpoint
    (epoch 15) primary, with the reason (best-epoch selection picked transient peaks).
- 3.5 Staged protocol and decision rules (Fig. 1)
  - Discovery seed 42; confirmation seeds 1337, 2026; recipe screening seeds 101–103;
    R1 confirmation seeds 201–203; one sealed test evaluation of all 36 runs.
  - Sign-based rules ("supported / contradicted / inconsistent") fixed before each stage.
  - Disclosures: rules were adopted after the discovery seed was seen; the stability rule
    cannot detect a consistent failure; normalization statistics include dev events.
- 3.6 Cost measurement
  - FLOPs (convolutions and matrix multiplications only), size, peak GPU memory, GPU latency
    FP32/FP16, ONNX Runtime CPU latency with 1 and 4 threads; warmup/timed/repetition
    protocol; shared-server caveat.
**Sources**: internal protocols; Daudt et al. 2018; Chen et al. 2022; Howard et al. 2019;
Salehi et al. 2017; Maier-Hein et al. 2024; Dehghani et al. 2022.
**Transition to next**: results follow the order of the research questions.

#### 4. Results (~2,000 words)
**Purpose**: report outcomes exactly as the pre-specified rules produced them, then describe.
**Serves sub-question**: 4.1 → RQ1; 4.2–4.4 → RQ2; 4.5 → RQ3.
**Content**:
- 4.1 The two components have no consistent effect (Table 3)
  - Size weighting and boundary package: "inconsistent" on validation under both recipes
    and on test under both recipes.
  - Contrasts on test are within about ±2.3 pp at 2–7% interior-small recall (n = 4,899).
- 4.2 Seed sensitivity and rank stability (Table 4, Fig. 2)
  - Validation: within-arm seed ranges up to 26.7 pp (FC-Siam-Diff under R1).
  - Test: 10/15 (original) and 9/15 (R1) arm pairs keep their order across seeds.
  - Order between families is stable (SmallFlood-CDNet < FC-Siam-Diff < BIT-SAR v2, one
    exception); order among A–D is not.
- 4.3 Variation concentrates in one event, but flood fraction does not explain it
  - Validation: Lumberton; arm A below 20% F1 in 42 of 45 epochs under R1; BIT-SAR v2 never
    below 20% in 90 epochs across six seeds.
  - Test: NovaKakhovka (highest flood fraction) is the most variable event in 8 of 12 groups;
    the pre-specified low-flood-fraction rule returns "does not generalize" (4 of 12).
  - Direction of failure: over-prediction (predicted positives above truth).
- 4.4 A recipe chosen on held-out training events did not transfer
  - R1 was best and most balanced on the internal dev split, but made arm A consistently
    worse on validation (about 59% event-macro F1 in all three seeds).
  - The stability rule labelled this "stable": a rule limitation, reported as such.
- 4.5 Accuracy and cost (Table 5, Fig. 3)
  - Test accuracy: BIT-SAR v2 73.9/74.2, FC-Siam-Diff 71.0/65.5, SmallFlood-CDNet arms
    55–62 (original/R1 seed means).
  - Cost: 0.35 G / 7.0 G / 21.4 G FLOPs; GPU 8.7 / 3.0 / 24.5 ms; CPU one thread
    10.7 / 81.4 / 350.9 ms; all under 20 MB.
  - Parameter count, FLOPs, GPU latency and CPU latency give four different rankings.
  - FP16 and ONNX conversion preserve validation metrics to within 0.18 pp.
**Sources**: the five result documents (see Evidence Map).
**Transition to next**: what these outcomes do and do not establish.

#### 5. Discussion (~850 words)
**Purpose**: interpret within the limits; give recommendations.
**Serves sub-question**: RQ1–RQ3 (interpretation).
**Content**:
- 5.1 What the negative result means: component gains seen in one seed were within seed
  variation; small floods remain essentially unsolved for all tested models.
- 5.2 What is not established: the cause of single-event over-prediction; why BIT-SAR v2 is
  robust (architecture, initialization and loss coefficients are confounded).
- 5.3 Recommendations for reporting: several seeds with ranges; per-event results; last
  checkpoint next to best; object-level metrics with denominators; several cost indicators
  on more than one device; rules fixed before looking.
- 5.4 Limitations: one dataset; three validation and three test events; three seeds per
  recipe; intensity only (no coherence); patch-level not scene-level; 15 epochs; shared
  server timing; no edge hardware; rules adopted after a discovery seed; the stability-rule
  loophole; bounded literature search.
**Sources**: result documents; Corley et al. 2024; Dehghani et al. 2022; Maier-Hein et al.
2024; Li et al. 2019; seed-variance sources.
**Transition to next**: short conclusion.

#### 6. Conclusion (~250 words)
**Purpose**: restate findings and the next bounded question.
**Serves sub-question**: RQ1–RQ3.
**Content**: three findings in one sentence each; future work stated as questions (what
makes BIT-SAR v2 robust; a second dataset; coherence input; measured edge deployment).
**Sources**: none new.

#### Back matter (not counted)
Acknowledgments and funding (user to supply); data and code availability (UrbanSARFloods is
public; code, protocols and result records in the project repository); disclosure of
interests; author contributions (CRediT) if the template allows; AI-assistance statement.

### Tables and figures (budget: 5 tables, 3 figures)

| Item | Content | Source document |
|---|---|---|
| Table 1 | Split: events, patches, positive fraction, interior-small counts (validation and test) | `heldout_test_results` §1; `direction_b_confirmation_protocol` §2 |
| Table 2 | Arms: parameters, loss, learning rate per recipe, initialization | `recipe_confirmation_protocol` §3; `efficiency_results` §2.1 |
| Table 3 | Sign-rule outcomes: 3 rules × {validation original, validation R1, test original, test R1} with per-seed contrasts | `direction_b_confirmation_results` §2; `recipe_confirmation_results` §2; `heldout_test_results` §2 |
| Table 4 | Test benchmark, last checkpoint: mean (range) of event-macro F1, pixel F1, component F1, interior-small recall, boundary F1, inner-band IoU | `heldout_test_results` §6.1 |
| Table 5 | Cost: parameters, FLOPs, size, GPU latency FP32/FP16, CPU latency 1/4 threads, peak memory | `efficiency_results` §2 |
| Fig. 1 | Staged protocol: seeds, recipes, splits, where each rule was fixed | protocols |
| Fig. 2 | Per-seed event-macro F1 by arm and recipe, validation and test (dot plot) | `recipe_confirmation_results` §4.1; `direction_b_confirmation_results` §4; `heldout_test_results` §3 |
| Fig. 3 | Test event-macro F1 versus CPU and GPU latency | `heldout_test_results` §6.1; `efficiency_results` §2 |

Per-seed and per-event tables that do not fit go to the repository, cited in the text.

### Evidence Map

**Internal evidence (results)**

| Section | Evidence | Document |
|---|---|---|
| 3.1 | Split sizes, events, denominators | `heldout_test_results_20261010.md` §1; `RESEARCH_HANDOFF.md` |
| 3.2–3.3 | Arms, recipes, pairing, determinism | `recipe_confirmation_protocol_v1_20261008.md` §3, §5; `direction_b_confirmation_protocol_v1_20261003.md` §4 |
| 3.5 | Rules and their provenance | the three protocols' "threshold provenance" sections |
| 4.1 | Component outcomes | `direction_b_confirmation_results_20261006.md` §2; `recipe_confirmation_results_20261009.md` §2; `heldout_test_results_20261010.md` §2 |
| 4.2 | Ranges and rank stability | `recipe_confirmation_results` §2, §4; `heldout_test_results` §3 |
| 4.3 | Lumberton and NovaKakhovka | `direction_b_confirmation_results` §6; `recipe_confirmation_results` §5; `heldout_test_results` §4–5 |
| 4.4 | Recipe selection and non-transfer | `recipe_stability_results_20261006.md` §2, §4; `recipe_confirmation_results` §3, §6 |
| 4.5, 3.6 | Cost and fidelity | `efficiency_results_20261010.md` §2–3 |
| 5.4 | Limitations | all closure documents' "not permitted / not established" sections |

**Literature (identity verified 2026-10-03 unless marked)**

| Source | Sections | Use | Status |
|---|---|---|---|
| Zhao et al. 2024 (UrbanSARFloods) | 1, 2.1, 3.1 | dataset; urban flood difficulty | verified; read at section level; full author list to complete |
| Bonafilia et al. 2020 (Sen1Floods11) | 1, 2.1 | SAR flood benchmark context | verified; abstract level |
| Bountos et al. 2024 (Kuro Siwo) | 2.1, 6 | larger dataset; future work | verified; author list to complete |
| Li, Martinis & Wieland 2019 | 1, 5.4 | coherence for urban floods | verified; abstract level |
| Saleh et al. 2024a, 2024b | 2.1–2.2 | Siamese SAR flood change detection | verified; abstract level |
| Daudt et al. 2018 (FC-Siam) | 2.2, 3.2 | baseline | verified |
| Chen, Qi & Shi 2022 (BIT) | 2.2, 3.2 | baseline origin | verified |
| Corley et al. 2024 | 1, 2.2, 5 | simple baselines stay competitive | verified; abstract level |
| Tomanič et al. 2026 | 2.2, 2.4 | benchmark with efficiency | verified preprint; modalities not checked |
| Salehi et al. 2017 (Tversky) | 2.3, 3.2 | loss | verified |
| Kofler et al. 2022; Rachmadi et al. 2024 | 2.3 | instance-aware losses | verified; abstract level |
| Cao et al. 2020 | 2.3 | small changed regions | verified; abstract level |
| Kervadec et al. 2019; Zhong et al. 2024 | 2.3 | boundary-aware methods | verified; abstract level |
| Maier-Hein et al. 2024 | 3.4, 5.3 | object-level metrics | verified |
| Howard et al. 2019 (MobileNetV3) | 3.2 | encoder | verified |
| Codegoni et al. 2022; Liu et al. 2024 | 2.4 | lightweight change detectors | verified preprints |
| Dehghani et al. 2022 | 1, 2.4, 3.6, 5.3 | parameters/FLOPs vs latency | verified |
| Mateo-Garcia et al. 2019, 2021; Inzerillo et al. 2025 | 2.4 | onboard flood/change detection | verified; abstract level |
| Seed-variance and reporting literature (candidates: studies of benchmark variance, random-seed effects and score-distribution reporting) | 1, 2.5, 5.3 | support for multi-seed reporting | **NOT yet verified: to be searched and identity-checked before drafting; none may be cited from memory** |
| Two "pending" SAR lightweight flood papers from the review | 2.4 | closest prior work | **pending: retrieve or drop** |

### Word Count Summary

| Section | Target words |
|---|---:|
| 1. Introduction | 800 |
| 2. Related Work | 750 |
| 3. Experimental Design | 1,300 |
| 4. Results | 2,000 |
| 5. Discussion | 850 |
| 6. Conclusion | 250 |
| **Total** | **5,950** |

Abstract, tables, figures, references and back matter are not counted. The binding constraint
is 12–15 pages in the LNNS template; if the draft runs long, cut 2.3–2.4 and move per-seed
detail to the repository before cutting 3.5 or 5.4.

## Open items before drafting

1. **Novelty check (required for 2.5 and the gap statement):** a targeted search for work
   citing UrbanSARFloods and for multi-seed or small-object evaluations in SAR flood change
   detection. The 2026-10-03 review states it is not a novelty clearance.
2. **Seed-variance sources:** find and verify; cite only what is verified.
3. **Full-text reading** of the sources used for specific claims (at least Zhao et al. 2024,
   Corley et al. 2024, Dehghani et al. 2022 and the two pending SAR papers).
4. **Anonymization:** the venue does not state its review model. Items to remove if the
   submission must be anonymous: repository URL, server and institution names, author-identifying
   acknowledgments.
5. **From the user:** author list and order, affiliations, corresponding author, funding and
   acknowledgments, choice of title, and whether code is released at submission time.
6. **Naming:** in the paper, rename "H0/H1/H3" and "arm" where a general AI reader may
   misread them (for example "recipe-transfer check", "component rules", "model variants").
