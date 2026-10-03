# Candidate R train eligibility audit

## Material Passport

Origin: academic-research-suite / experiment-agent diagnostic continuation.
Input evidence: user reports Candidate R loss tests 19 passed on CUDA server.
Scope: read train NPY labels/valid masks, no model, optimizer, test/validation arrays,
SAR arrays or component caches. Only new report and patch-count artifacts written.
No approval to start A-control/R training is implied.

Uses exactly build_local_supervision from Candidate R loss, area<=35, 8-connectivity,
interior eligibility and radius3 background. Counts eligible objects/pixels, eligible
patches, empty rings and small clipped objects, globally/per event. Simulates all15
epoch batch schedules with batch8 and independent seed42 generator using metadata
only. Generator consumption matches factorial DataLoader. Compares hashes against
historical A if its summary exists; mismatch stops. This simulation does not run15
epochs of training and does not measure model/gradient behavior.

From server project root, with existing .venv:

```bash
python scripts/audit_candidate_r_eligibility.py --self-test
# Expected 3 passed; then inside tmux:
set -o pipefail
mkdir -p artifacts/candidate_r_eligibility
python -u scripts/audit_candidate_r_eligibility.py 2>&1 | tee "artifacts/candidate_r_eligibility/scan_$(date -u +%Y%m%dT%H%M%SZ).log"
```

Reports every500 train patches. Expected19166. New UTC directory contains report.json
and patch_counts.json; existing output directory refused. Metadata hashes frozen and
checked before/after. Do not modify NPYs while scan runs; no content-level concurrent
write protection. Failure stops with no automatic retry. Success marker:
CANDIDATE R ELIGIBILITY AUDIT COMPLETE.

Local 3 tests cover label/valid-only loading, unchanged input files, test-split rejection,
known eligibility counts and deterministic metadata-only schedule including empty batches.
The loss module's separate tests already compare eligibility with the evaluation metric.
After results: assess how often auxiliary signal is present, not whether it improves
recall. Do not change sampler/threshold/lambda based on exposure counts without a new
explicit protocol decision. No training runner integration in this package.
