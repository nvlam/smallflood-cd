# Evaluation metric contract

Every metric below corresponds to a frozen methodology claim. All calculations accept
the same effective valid mask, including uncertain-border exclusions.

## Pixel metrics

For valid pixels, confusion counts are (TP,FP,FN,TN\). The reported measures are

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},
\]

\[
F_1=\frac{2TP}{2TP+FP+FN},\qquad
\mathrm{IoU}=\frac{TP}{TP+FP+FN}.
\]

These are the standard aggregate comparisons required for FC-Siam-Diff, BIT, and the
proposed method. They do not by themselves support the small-change contribution.

## Component-level metrics

Predicted and reference foreground masks are partitioned with the same frozen
connectivity used during training. Their pairwise IoU matrix is computed, and candidate
pairs above the frozen match threshold are assigned greedily in descending IoU order,
with one prediction and one target per match.

\[
P_c=\frac{N_{match}}{N_{pred}},\qquad
R_c=\frac{N_{match}}{N_{gt}},\qquad
F_{1,c}=\frac{2N_{match}}{N_{pred}+N_{gt}}.
\]

Mean matched IoU measures the localization quality of successfully detected components.
One-to-one matching prevents a single merged prediction from receiving credit for
multiple separate target regions.

## Small-component recall

The small-area threshold (A_{small}\) is derived from training labels and frozen before
test evaluation. A small reference component is detected only if it participates in a
valid one-to-one component match:

\[
R_{small}=\frac{N_{matched,\,a_k\le A_{small}}}{N_{gt,\,a_k\le A_{small}}}.
\]

This is the direct evaluation counterpart of component-size-aware weighting and the
approved missed-small-change research question.

## Boundary IoU and F-score

Boundary maps use the same morphological-gradient definition as boundary supervision.
Exact boundary IoU is

\[
\mathrm{BIoU}=\frac{|E_p\cap E_g|}{|E_p\cup E_g|}.
\]

Because one-pixel offsets can be excessive at raster boundaries, BF1 uses a frozen
tolerance (\tau\). Predicted boundary pixels are correct when they fall within the
dilation of the target edge, and conversely for recall:

\[
P_b=\frac{|E_p\cap\operatorname{Dilate}_\tau(E_g)|}{|E_p|},\qquad
R_b=\frac{|E_g\cap\operatorname{Dilate}_\tau(E_p)|}{|E_g|},
\]

\[
BF_1=\frac{2P_bR_b}{P_b+R_b}.
\]

Boundary IoU supplies strict overlap evidence; BF1 supplies tolerance-aware localization
evidence. Together they test the thin boundary-refinement head without relying on visual
examples.
