# Simulated pre-submission review: ICEIR 2026 draft

## Material Passport

Origin: ARS `academic-paper-reviewer`, `full` mode requested by the user on 2026-10-10
("chạy phản biện giả lập"). Manuscript reviewed: `paper/iceir2026/main.tex` at commit `213c3ee`
(13 pages). The manuscript was **not modified** by this review.

**Execution provenance (read before relying on this report).** This review was produced in
a **single model context by one model**, not by five separately dispatched reviewer agents.
Consequences:
- role separation: persona only; invocation contexts were **not** fresh per seat; peer
  outputs were visible across seats; model family identical for all seats;
- the sprint-contract two-call protocol (paper-blind Phase 1, paper-visible Phase 2), the
  conformance and panel-synthesis checkers and the `review-panel-provenance` artifact were
  **not run**;
- the reviewer also wrote the draft and holds the project's result records, so this is a
  self-review with the corresponding blind spots;
- calibration status: `NOT_CALIBRATED`; `criteria_binding_unavailable` (no venue criteria
  registry bound; no venue-alignment claim).
The five perspectives below are therefore angles of one reading, **not independent
reviews**. A full five-agent panel can be run separately if the authors want it.

Field analysis: primary discipline remote sensing (SAR) with deep learning; secondary
machine-learning evaluation methodology; empirical, quantitative; conference paper for a
general intelligent-computing venue; maturity: complete first draft with placeholders.

## Reviewer configuration (not pre-confirmed by the user)

| Seat | Perspective |
|---|---|
| Journal-Fit Reviewer | PC member of a general AI conference with LNNS proceedings: fit, originality, clarity for a non-remote-sensing reader |
| Reviewer 1 (Methodology) | Experimental design and evaluation methodology in deep learning: seeds, decision rules, confounds, reproducibility |
| Reviewer 2 (Domain) | SAR flood mapping and change detection: data handling, baselines, literature |
| Reviewer 3 (Perspective) | ML efficiency and deployment: cost measurement and practical implications |
| Devil's Advocate | Strongest counter-argument to the central claims |

## 1. Journal-Fit Reviewer

**Strengths.** The paper is honest about what it is: a negative result with a benchmark. The
research questions are explicit, the limits are stated, and the test set was opened once
under a written plan. This is uncommon and is the paper's main asset.

**Weaknesses.**
- **F1 (Major). Fit and framing for this audience.** ICEIR's topics are intelligent
  computing and information retrieval. The introduction assumes the reader knows SAR,
  backscatter, VH/VV and coherence. *Where:* Sect. 1, first two paragraphs. *Fix:* add two
  sentences that explain the task in general terms (two radar images, before and during a
  flood; output a flood mask) and lead with the machine-learning lesson (single-seed
  comparisons, event-level evaluation, cost indicators).
- **F2 (Major). Contribution 4 is not backed by the manuscript.** *Where:* Sect. 1,
  fourth bullet ("Protocols, decision rules, code and per-run records"). There is no
  availability statement and no URL. *Fix:* add a data and code availability statement, or
  remove the bullet.
- **F3 (Minor). Placeholders.** Author block, running head, acknowledgments and the
  AI-assistance sentence are unfinished.
- **F4 (Minor). No visual example.** A general reader never sees a SAR pair, a label or a
  prediction. One qualitative figure (for instance a Lumberton patch with over-prediction)
  would carry Sect. 4.3 better than text.

**Recommendation signal:** major revision before submission; no fatal flaw.

## 2. Reviewer 1 (Methodology)

**Strengths.** Paired initialization and batch order within a seed; deterministic training;
rules written before each stage and disclosed as post-discovery; last checkpoint justified;
test protocol with fixed checkpoint list and hash checks.

**Weaknesses.**
- **M1 (Major). The sign rule is strict, and the text does not show how the data sit
  relative to it.** "Supported" or "contradicted" needs six contrasts of one sign; with
  noise, "inconsistent" is the likely outcome whatever the truth. On test, 8 of 12
  size-weighting contrasts are positive (mean about +0.7 pp) and 9 of 12 boundary contrasts
  are positive (mean about +0.4 pp). *Where:* Sect. 4.1, Table 3 (ranges only). *Fix:*
  report the per-seed contrasts, the count of positive signs and the mean; say plainly that
  any effect on test is at most about 2 pp at 2–7% recall, and that its sign is not
  consistent. Do not change the rule or its outcome.
- **M2 (Major). A fixed 0.5 threshold cannot separate ranking from calibration.** The
  single-event failure is described as over-prediction. Without a threshold-free measure
  (average precision or a precision–recall curve) on validation or test, the reader cannot
  tell whether the models rank pixels badly or are merely miscalibrated on that event.
  *Where:* Sects. 3.4 and 4.3. *Fix:* state this as a limitation and as an untested
  alternative explanation. (Average precision exists only for the phase-1 development
  split.)
- **M3 (Major). Unequal tuning across families.** BIT-SAR v2 used its own learning rate from
  the start; R1 was tuned for variant A only; the baselines differ in loss coefficients and
  initialization. *Where:* Sect. 3.2 states the caveat in one sentence. *Fix:* repeat it
  where the family ranking is interpreted (Sects. 4.2, 4.5 and the conclusion).
- **M4 (Major). The training regime may be weak.** Fifteen epochs, no augmentation, training
  from scratch. Interior-small recall of 2–7% and event-macro F1 of 55–74% could reflect an
  under-trained regime in which no loss modification can show an effect. *Where:* Sect. 3.3;
  the caveat appears only in Sect. 5. *Fix:* give the evidence that training converged
  (loss values are in the result records), and state the regime as a condition of the
  negative result in the abstract and conclusion.
- **M5 (Minor).** Table 3 caption does not say that two of the three validation rules under
  the original recipe used the best checkpoint; the text does (Sect. 3.5).
- **M6 (Minor).** With three seeds, "range" is the right summary, but say so once and state
  that no variance estimate is attempted.
- **M7 (Minor).** The internal development events are not named, and the number of runs and
  the compute used are not given.

**Recommendation signal:** major revision; all points can be handled in text.

## 3. Reviewer 2 (Domain)

**Strengths.** Event-held-out split; explicit statement that the split is not the official
one; object-level metrics with denominators; intensity-only limitation acknowledged.

**Weaknesses.**
- **D1 (Major). SmallFlood-CDNet is not described well enough to reproduce.** It is the
  authors' own model and has no prior publication. *Where:* Sect. 3.2, one sentence. *Fix:*
  add the decoder and fusion design and the boundary head in a short paragraph or a small
  architecture figure; the same for what was changed from BIT to obtain BIT-SAR v2.
- **D2 (Major). No external anchor for model quality.** The paper does not relate its
  numbers to any published result, and says they are not comparable with the dataset paper.
  A reader cannot tell whether the baselines are reasonably trained. *Fix:* say what the
  dataset paper reports qualitatively (urban flood remains hard) and why a numeric
  comparison is not possible (different split, binary target, patch size).
- **D3 (Minor). Label merging.** Flooded open areas and flooded urban areas are merged into
  one class, yet the motivation is urban flooding. *Where:* Sect. 3.1. *Fix:* state that
  urban-specific performance is not measured, and that "small" refers to component size, not
  to the urban class.
- **D4 (Minor). Literature at abstract level.** Several related-work statements rest on
  abstracts; five bibliography entries are incomplete (marked CHECK). *Fix:* complete the
  entries; keep claims generic where the full text was not read.
- **D5 (Minor).** Table 4 omits inner-band IoU although the boundary rule in Table 3 is
  defined on it.

**Recommendation signal:** major revision.

## 4. Reviewer 3 (Perspective: efficiency and deployment)

**Strengths.** Several cost indicators on two devices; FLOP-counter scope disclosed; FP16 and
ONNX fidelity checked; no edge-hardware claim.

**Weaknesses.**
- **P1 (Major). The cost section invites a deployment reading it cannot support.** A
  server GPU and a server CPU are not the edge. *Where:* Sect. 4.5 and Fig. 2. *Fix:* one
  explicit sentence that the CPU point is a server CPU and that the result is "indicators
  disagree", not "model X suits the edge".
- **P2 (Minor). Why does the low-FLOP model run slower on the GPU?** The paper reports it
  without comment. *Fix:* say it was not investigated; do not offer a cause that was not
  tested.
- **P3 (Minor). Timing noise.** Mentioned in one clause. *Fix:* give the spread of the three
  repetition medians for at least one row, or point to the released records.
- **P4 (Minor).** Batch-8 throughput was measured but is not reported; either add one
  sentence or drop the claim that half precision "did not reduce batch-1 latency" from
  carrying a general message.

**Recommendation signal:** minor revision for this part.

## 5. Devil's Advocate

**Strongest counter-argument.** The paper's title and abstract announce a "controlled
negative result", but the design cannot distinguish "no effect" from "a small positive
effect". The decision rule demands the same sign in six paired contrasts; a single contrast
of −0.08 pp turns the R1 test outcome into "inconsistent" although five of six contrasts are
positive. Counted over both recipes, size weighting raised test interior-small recall in 8 of
12 paired contrasts. At the same time the training regime (15 epochs, no augmentation, from
scratch) leaves every model at 2–7% recall on those components, a floor at which a loss
re-weighting has little to act on. A reader could fairly conclude the opposite of the
paper's emphasis: that the experiment was under-powered and under-trained, that the rule was
built to return "inconsistent", and that the only firm findings are the family ranking and
the cost table. The seed-sensitivity message is real, but it then argues against the
strength of the negative claim rather than for it.

**Issues.**
- **DA-1 (CRITICAL; validated). Wording of the central claim.** "No consistent effect" is
  accurate under the rule, but "does not help" (Introduction, title of the pilot narrative,
  Discussion) over-reads it. *Fix:* phrase the claim as "no reliable effect was detected; any
  effect on test is at most about 2 pp and of inconsistent sign", show the per-seed
  contrasts, and name the training regime as a condition. This is a wording and reporting
  change; no new experiment is needed.
- **DA-2 (MAJOR).** Interior-small recall on test is dominated by one event (Nova Kakhovka,
  89% of components). The "sealed test" evidence for RQ1 is effectively one event.
  *Where:* Sect. 3.1 states the share; Sect. 4.1 does not draw the consequence.
- **DA-3 (MAJOR).** "Seed sensitivity concentrates in one event" is shown for the smallest
  family at a fixed threshold; see M2. The paper should not present it as a property of the
  events.
- **DA-4 (MINOR).** Sect. 4.4's account of the stability-rule loophole is candid but long
  for a 13-page paper and may read as internal history.

**Alternative explanations not ruled out:** threshold miscalibration under event shift;
under-training; the Tversky false-negative weight (0.7) encouraging over-prediction; lack of
augmentation; label characteristics of the affected events.

**Observations (not defects):** the single-use test protocol and the disclosed post-hoc
status of several analyses are exemplary and should be kept.

## Editorial synthesis

**Consensus (all perspectives):** the study is carefully run and honestly reported; the
weaknesses are in reporting and wording, not in the experiments. No requested change
requires a new training run or another look at the test set.

**DA CRITICAL adjudication:** DA-1 is **validated**. The data support "no reliable effect of
practical size", not "no effect". It blocks an "accept as is" outcome until the wording and
the per-seed contrasts are fixed.

**Decision: major revision before submission** (in the sense of substantial text changes;
feasible within the remaining days).

## Revision roadmap (source order; triage is the authors' decision)

| ID | Item | Severity | Needs new experiment? |
|---|---|---|---|
| DA-1 / M1 | Reword the central claim; add per-seed contrasts, sign counts and means | Critical | No (numbers are in the test record) |
| M4 | State the training regime as a condition; give convergence evidence | Major | No |
| M2 / DA-3 | Add the threshold/calibration limitation and alternative explanations | Major | No |
| M3 | Repeat the unequal-tuning caveat where rankings are interpreted | Major | No |
| D1 | Describe SmallFlood-CDNet and BIT-SAR v2 adequately | Major | No (authors' knowledge of the code) |
| F2 | Availability statement or remove contribution 4 | Major | No |
| F1 | Rewrite the opening for a general AI reader | Major | No |
| DA-2 | Note that test evidence for RQ1 rests mainly on one event | Major | No |
| D2 | Explain why no numeric comparison with published results is possible | Major | No |
| P1 | Guard the cost section against a deployment reading | Major | No |
| F4 | One qualitative figure | Minor | No new run; needs existing validation preview images from the server |
| D3, D5, M5–M7, P2–P4, DA-4, F3, D4 | Minor items as listed above | Minor | No |

## Attachment: Acronym Check (advisory, #849)

The acronym check did not run (the script accepts Markdown or plain text, not LaTeX). By
inspection, these are used without expansion and should be defined at first use: BCE, IoU
(in table captions), FLOPs, ONNX, FP32/FP16, VH/VV, BIT.
