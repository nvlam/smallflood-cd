# Validation / checkpoint protocol v2

Status: implementation tested locally; remote GPU verification pending.
Scope: user-approved step 1 after engineering pilot review. No new training or test evaluation.

- Applies equally to Proposed, FC-Siam-Diff and compact BIT.
- Use validation only, threshold 0.5, unshifted predictions without morphology.
- Exclude invalid pixels (and supplied uncertain masks with configured dilation).
- Sum integer TP/FP/FN/TN over patches within each event before computing metrics.
- Select maximum arithmetic mean of defined event F1 values. Tie: earlier epoch.
- Zero denominator means null; report the number of defined events for each macro metric.
- Report global pooled-pixel precision/recall/F1/IoU alongside event macro and per-event metrics.
- Reject absent/nonfinite selection scores; never fall back to legacy patch-average/composite scores.
- Keep `best_composite.ckpt` as a legacy filename only. It no longer denotes composite selection.
- Save `selection_protocol.json` and protocol metadata in newly created checkpoints.
- New fits must use new directories; no implicit resume or overwrite.

Earlier pilot checkpoints, logs and retrospective validation reports remain unchanged.
This protocol was selected after observing exploratory pilot validation results; disclose this history.
Do not label the earlier pilot as prospectively run under v2.

Implementation: engine/validator.py, engine/trainer.py, engine/experiment.py and scripts/next_steps.py.
The full experiment runner's final test/component report still uses legacy collection; it is NOT
approved for the 135-run study yet. Step 2 must address boundary/component definitions and reporting.
Band semantics and GPU reproducibility remain gates before official experiments.
Unit tests include pooled-vs-macro behavior, undefined cases, valid masks, batch-size invariance,
and checkpoint selection when global F1 and the old composite score prefer a different epoch.
