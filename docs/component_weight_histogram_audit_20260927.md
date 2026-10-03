# Train component-weight histogram audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: validate; Verification Status: ANALYZED
- Date: 2026-09-27
- Source: user-provided /Users/nguyenvulam/Downloads/component_stats_v1.json
- SHA256: 525e4e1354e991983ebd2e3d3f68caad3f9894308a6eae46256a74405c9f2247
- No training, model changes, raw mask scan, or test access.

Hash matches component statistics recorded in prior validation reports. Metadata:
train-fitted, 8-connectivity, 256-patch clipped including tile edges; reference area
65, small threshold 35, requested small quantile 0.25. Histogram entries have positive
integer areas and counts. Threshold ties explain why <=35 comprises 25.70%, not 25%.

## Exact histogram aggregates

52,516 components; 23,436,067 foreground pixels represented in histogram.
Weights reconstructed from current pilot formula:
w(A)=1+min(4,sqrt(65/(A+1e-6))). Background has weight 1 but is absent from histogram.

| Area | Components | Component share | Foreground pixels | Pixel share | Weighted foreground mass share |
|---|---:|---:|---:|---:|---:|
| 1–4 | 5,120 | 9.7494% | 10,136 | 0.04325% | 0.17704% |
| 5–35 | 8,377 | 15.9513% | 182,160 | 0.77726% | 1.69492% |
| 36–65 | 12,918 | 24.5982% | 630,324 | 2.68955% | 4.73366% |
| >65 | 26,101 | 49.7010% | 22,613,447 | 96.48994% | 93.39438% |

Small <=35 totals: 13,497 components, 192,296 pixels, 25.7007% of components,
0.82051% of foreground pixels, 1.87195% of weighted foreground mass.
Mean weight across small foreground pixels: 2.78673.
Mean weight across all components: 2.29619; across foreground pixels: 1.22148.
Weights range 1.03153–5. The cap applies to enhancement, so final cap is 5.

## Interpretation and limitations

The weighting enhances small-region pixels, but does not equalize total per-object
contribution. Small-region share of positive weighted mass rises from 0.82% to 1.87%;
regions >65 pixels still account for 93.39%. This supports checking contribution
balance, not concluding that loss weighting caused pilot performance.

Weighted mass is sum(A*n_A*w_A), NOT measured loss or gradient contribution. Actual
contributions also depend on predictions, valid/uncertain masks, batch composition,
BCE normalization, Tversky coupling and background pixels. Histogram includes clipped
components; full scene objects and interior-only evaluation populations differ.
Raw masks were not supplied, so valid-mask handling cannot be verified here.

Fallacy scan 11/11: aggregation noted (component vs pixel denominators); ecological
claims excluded; selection restriction noted (patch-clipped train objects); collider
and reverse causality not applicable; base rates explicit; no regression-to-mean
claim; histogram coverage not equated to independent raw-data completeness; no
significance cherry-picking; protocol not retuned; no causal claim about pilot scores.
No p-values, confidence intervals, independent events or repeated seeds inferred.

## Next diagnostic gate

Read-only train-mask audit: compare cached-area definitions with valid-masked labels,
count positive labels in invalid pixels, count supervised boundary pixels adjacent to
invalid pixels, and collect actual weight distributions. Keep checkpoints, loss,
thresholds and test split untouched. Do not raise weights before this evidence.
