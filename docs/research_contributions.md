# Research contribution modules

This document freezes the mathematical-to-code mapping for the five approved
methodological contributions. Infrastructure and baselines are outside its scope.

## 1. Cached connected components

For a binary target mask (Y\), changed pixels are partitioned into maximal connected
sets (C_1,\ldots,C_K\) under fixed 4- or 8-neighbor connectivity. The component map is

\[
L_i = k \quad \text{iff} \quad i \in C_k,
\]

with (L_i=0\) for unchanged pixels. `ConnectedComponentCache` hashes the mask bytes,
shape, connectivity, and cache version. The resulting label map and areas are stored in
a compressed, content-addressed artifact. Consequently, the same target and definition
produce the same cache entry, and changed connectivity produces a different entry.

This implements the methodology requirement to precompute component identities instead
of repeating connected-component analysis during every epoch.

## 2. Size-aware weights

For component area (a_k=|C_k|\), training-only reference area (A_{ref}\), and pixel
(i\in C_k\),

\[
w_i = 1 + \alpha\min\left(w_{max},
\left(\frac{A_{ref}}{a_k+\epsilon}\right)^\gamma\right).
\]

Unchanged pixels receive weight 1. `reference_component_area` computes (A_{ref}\) as
the median area over training maps only. `component_weight_map` maps the cached IDs to
pixel weights. The square-root default (\gamma=0.5\) and clipping prevent tiny noisy
labels from dominating while still increasing the contribution of small changes.

## 3. Size-aware Tversky loss

With valid mask (V\), probability (p_i\), and size weights applied only to positives,

\[
TP_w=\sum_i V_iw_iY_ip_i,\qquad
FP=\sum_iV_i(1-Y_i)p_i,
\]

\[
FN_w=\sum_iV_iw_iY_i(1-p_i).
\]

The loss is

\[
\mathcal L_{SAT}=1-
\frac{TP_w+\epsilon}{TP_w+\beta FP+\delta FN_w+\epsilon}.
\]

`SizeAwareTverskyLoss` implements this formula directly. The default
(\beta=0.3,\delta=0.7\) penalizes missed changed pixels more strongly, matching the
approved missed-small-change priority without globally overweighting negative pixels.

## 4. Boundary supervision

The target edge is the morphological gradient

\[
E=\operatorname{Dilate}_r(Y)-\operatorname{Erode}_r(Y).
\]

`build_boundary_supervision` constructs this target and its valid supervision mask.
The composite objective applies equal BCE and Dice contributions to the thin edge target.
This directly supervises localization while the boundary head remains a small residual
correction rather than a second segmentation decoder.

## 5. Uncertain-border masking

Given an author- or preprocessing-supplied uncertain mask (U\), a safety band is

\[
U'=\operatorname{Dilate}_{r_u}(U).
\]

The effective valid mask is

\[
V'=V\odot(1-U').
\]

`apply_uncertain_border_mask` applies this operation before change and boundary losses.
Thus uncertain labels cannot be rewarded as precise small components or precise edges.
The same mask contract is available to evaluation, preventing training/evaluation
definitions from diverging.
