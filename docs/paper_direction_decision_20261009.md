# Paper direction decision: A + C

## Material Passport

Decision: user reply "Chọn hướng A + C", 2026-10-09, after
`docs/recipe_confirmation_results_20261009.md` closed phase 2.
Status: **direction chosen.** No experiment, test-set access or code change is approved by
this record. Efficiency measurement (A3) and the single held-out test evaluation (A4) each
need their own protocol and approval.

## 1. Chosen framing

The paper combines two of the five options listed on 2026-10-09:

- **A. Controlled negative result with seed and event sensitivity.** Under frozen,
  reproducible protocols, size weighting and the boundary package show no consistent effect
  across six seeds and two training recipes. Seed variation concentrates on one
  low-flood-fraction validation event (Lumberton) and can reverse method rankings.
- **C. Lightweight benchmark with a full cost set.** Accuracy and cost of 0.5M–3.5M-parameter
  SAR change detectors under one controlled protocol, using the "edge-oriented efficiency,
  deployment deferred" framing of `docs/literature_review_direction_b_20261003.md` §8.

Not chosen: B (evaluation-methodology paper across datasets), D (mechanism study of
BIT-SAR v2's robustness; may be revisited as a separate protocol), E (original method
paper; not supported by the evidence).

## 2. Evidence already in hand

- `docs/direction_b_confirmation_results_20261006.md`: old recipe, seeds 42/1337/2026, all
  three rules inconsistent.
- `docs/recipe_stability_results_20261006.md`: phase 1, R1 selected on an internal dev split.
- `docs/recipe_confirmation_results_20261009.md`: phase 2, R1, seeds 201–203, H1 and H3
  inconsistent; H0 "stable" only through a consistent Lumberton failure.

## 3. Claims the paper must not make

- That size weighting or the boundary package improves small-flood or boundary metrics.
- That R1, or any recipe, solves the instability.
- That BIT-SAR v2's robustness comes from its architecture.
- Real-time or onboard deployment on Jetson Orin Nano (unmeasured).
- Any statement from the test set beyond the one evaluation that A4 will pre-specify.

## 4. Work remaining (each step needs its own approval)

1. **A4 protocol:** which checkpoints are evaluated once on the sealed test set, with which
   metrics and reporting rules, fixed before any test array is opened.
2. **A3 protocol:** the cost set (parameters, MACs/FLOPs, serialized size, peak memory,
   latency and throughput with a stated warmup/repetition protocol; FP16 fidelity if
   TensorRT is used).
3. **Paper outline and drafting,** for example with ARS `academic-paper` (plan or outline
   mode), from the closure documents above.
