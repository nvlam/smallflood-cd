# Read-only Proposed train-mask audit

## Material Passport

Origin: academic-research-suite, experiment diagnostic. User approved train-only audit.
No model, checkpoint, optimizer, SAR array, connected-component cache or evaluation
array is opened. Manifest metadata is read to select train rows. Only output report
and optional tee log are written. Existing output directories are refused.

Run from server project root:

```bash
python scripts/audit_proposed_train_masks.py --self-test
# Expected: 4 passed; then, inside tmux:
set -o pipefail
mkdir -p artifacts/proposed_train_mask_audit
python -u scripts/audit_proposed_train_masks.py 2>&1 | tee "artifacts/proposed_train_mask_audit/scan_$(date -u +%Y%m%dT%H%M%SZ).log"
```

Scans every train label and valid NPY (256x256), reports every 500 patches, CPU only.
Expected train count from prior pilot: 19166. No timing promise. Failures stop; no
automatic retries. SciPy already used by existing object/boundary review is required.
Self-test compares 8-connected area histograms with project cache labeling and uses
inaccessible train SAR/validation/test paths to test read scope; input hashes preserved.

Report includes raw and valid-masked component histograms, comparison with stored
train histogram, positive labels in invalid pixels, weight differences after masking,
boundary-supervised pixels adjacent to invalid regions (r1), and supervised boundary
pixels outside evaluation interior (r3, also including patch exterior). The last two
are distinct: counts indicate exposure, not proof of erroneous labels or causality.
Per-event and global totals are included. Weights use frozen R=65, alpha=1, gamma=.5,
enhancement cap=4, small area<=35. Foreground weighted mass is not a gradient/loss
contribution measurement. Uncertain-mask inputs are rejected instead of ignored.
Metadata hashes checked before/after; NPY content consistency during concurrent edits
is not guaranteed, so do not modify prepared data while scanning.

Success marker: TRAIN MASK AUDIT COMPLETE. Send report.json from printed output path.
No training or test evaluation should be launched based only on these diagnostics.
