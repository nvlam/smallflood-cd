# Integrative literature review for direction (b): lightweight SAR flood change detection without edge hardware

## Material Passport

Origin: ARS `deep-research` lit-review mode, Claude Code session, 2026-10-03.
Review form: **integrative review**, chosen by the user. This review supports the existing
SmallFlood-CD project and paper. It is not a standalone systematic review.
Status: first pass. Search coverage is bounded (see Method). Most sources were read at
abstract or landing-page level, not as full text.
Scope rule from the user: do not redesign the project and do not change frozen decisions.
No experiments were run, and no configuration or source file was changed.
Project context used: `RESEARCH_HANDOFF.md`, `docs/candidate_r_validation_review_20261003.md`,
`docs/proposed_factorial_validation_audit_20260928.md`.

Every statement below carries one of four labels:

- **[E] Established:** reported in a cited source.
- **[G] Gap:** absence or weakness found within the searches listed here. Gaps are bounded
  by this search and are not claims about the whole literature.
- **[F] Frozen:** an existing SmallFlood-CD decision, unchanged by this review.
- **[H] Hypothesis or proposal:** suggested by the review. Not adopted; each needs user approval.

## 1. Method

**Question.** What does the literature establish for these eight themes, and where is the
gap that motivates SmallFlood-CD's frozen design?

1. SAR / Sentinel-1 urban flood extent and change detection
2. Bitemporal / Siamese change detection
3. Small changed regions
4. Size- or instance-aware losses
5. Boundary-aware segmentation and change detection
6. Lightweight change-detection architectures
7. Edge deployment (ONNX, TensorRT, Jetson-class)
8. The resulting research gap

Direction (b) adds a ninth question: **how should efficiency be argued when no Jetson
device is available?**

**Search.**
- Web search (general engine plus arXiv, CVF, PMLR, MDPI and publisher landing pages),
  run on 2026-10-03.
- 31 queries in total. Query families:
  - dataset names (UrbanSARFloods, Sen1Floods11, Kuro Siwo, S1GFloods);
  - "SAR flood change detection Siamese";
  - "urban flood SAR double bounce coherence";
  - "Tversky / blob / ICI / boundary loss";
  - "small changed regions remote sensing";
  - "boundary-aware change detection";
  - "lightweight change detection parameters FLOPs";
  - "onboard flood detection Jetson PhiSat";
  - "efficiency misnomer FLOPs latency";
  - "TensorRT FP16 segmentation accuracy";
  - "change detection reality check / benchmark";
  - a citation-forward search on UrbanSARFloods.
- Not searched: Scopus, Web of Science or IEEE Xplore with Boolean strings, and no
  systematic citation chaining. A systematic or scoping extension would need these.

**Inclusion criteria.**
- Peer-reviewed papers or established preprints that bear directly on one of the themes.
- Bibliographic identity confirmed on an official or indexing page during this session.

**Exclusion criteria.**
- Sources whose identity or authors could not be confirmed. These are listed in §9 as
  pending and are not used as evidence.
- Optical-only change-detection papers, unless they speak to a method or evaluation issue
  that transfers to this project.

**Read scope.** Abstract or landing page for most sources. UrbanSARFloods was read at
section level through the arXiv HTML. No method-level weaknesses are claimed for sources
read only at abstract level.

## 2. Theme 1: SAR urban flood mapping

- **[E]** SAR is the standard sensor for flood mapping because it images through cloud,
  day or night. Deep models trained on Sentinel-1 can beat threshold methods when the
  training labels include floodwater specifically (Bonafilia et al., 2020).
- **[E]** Urban flooding is physically different from open-area flooding. Partly
  submerged buildings raise backscatter through double bounce, where open water lowers
  it. Adding multi-temporal interferometric coherence makes urban inundation maps more
  reliable. Multi-temporal intensity was the most important input in a TerraSAR-X case
  study of Hurricane Harvey (Li et al., 2019).
- **[E]** UrbanSARFloods (Zhao et al., 2024):
  - 8,879 Sentinel-1 chips of 512×512 pixels at 20 m, 18 events.
  - Labels: non-flooded (NF), flooded open (FO), flooded urban (FU).
  - Inputs are pre- and co-event intensity plus coherence, framed as semantic
    segmentation of stacked inputs.
  - Nine CNN segmentation baselines trained with weighted cross-entropy.
  - FO F1 ranged roughly 0.51–0.77 across test sites; FU F1 stayed **below 0.1** for all
    models (Section 4.3, Tables 3–4).
  - The authors conclude that weighted cross-entropy and ImageNet transfer learning do not
    overcome class imbalance and small training sets.
- **[E]** Larger SAR flood benchmarks exist. Kuro Siwo has 43 events, multi-temporal GRD
  and SLC products, and the BlackBench baselines (Bountos et al., 2024). Sen1Floods11 has
  11 events of single-date chips (Bonafilia et al., 2020).
- **[F]** SmallFlood-CD keeps a different contract:
  - Binary target (FO and FU merged), intensity only (VH, VV at two dates), no coherence.
  - 256×256 patches and an event-held-out repartition.
  - Its numbers are therefore **not directly comparable** with the UrbanSARFloods paper's
    three-class tables.
- **[G]** Within these searches, no paper was found that reports FU-specific recall for a
  binary flood change detector on UrbanSARFloods. Nor was any paper found that evaluates
  UrbanSARFloods as a bitemporal change-detection task with object-level metrics.

## 3. Theme 2: bitemporal and Siamese change detection

- **[E]** FC-Siam-diff, a fully convolutional Siamese network that decodes absolute feature
  differences, was introduced as a fast baseline for coregistered image pairs
  (Daudt et al., 2018).
- **[E]** BIT compresses bitemporal features into a few semantic tokens with a transformer.
  Its authors report better accuracy than a convolutional baseline at roughly three times
  lower compute and parameters (Chen et al., 2022).
- **[E]** SAR flood change detection has adopted Siamese designs:
  - DAM-Net: a Siamese ViT with temporal differential fusion, on the S1GFloods benchmark
    of 5,360 pairs and 46 events (Saleh et al., 2024a).
  - SemT-Former: a semantic-token transformer, Khartoum case study (Saleh et al., 2024b).
  - These papers report pixel-level OA, F1 and IoU.
- **[E]** "Reality check" studies find that simple U-Net-style baselines remain top
  performers on standard optical change-detection benchmarks once training is controlled
  (Corley et al., 2024). A 2026 benchmark of ten architectures on ten datasets reports
  that well-optimized Siamese U-Nets often beat newer models once efficiency is counted
  (Tomanič et al., 2026, preprint).
- **[F]** The project's pilot ranking agrees with this pattern but is single-seed
  validation evidence only. FC-Siam-Diff (0.49M parameters) led on event F1, interior-small
  recall and inner-band IoU; BIT-SAR v2 has 3.49M parameters.
- **[G]** The "simple baseline is competitive" result is documented for optical benchmarks.
  Within these searches it was not found tested on SAR urban flood change detection with
  small-region metrics.

## 4. Theme 3: small changed regions

- **[E]** Small changed regions are hard for two reasons: models optimized for overall
  accuracy favour large changes, and repeated pooling discards small-object detail.
  - Si-HRNet keeps high-resolution branches throughout a Siamese network and pairs weighted
    BCE with an inverse-volume-weighted generalized Dice loss (Cao et al., 2020).
- **[E]** Pixel-level overlap metrics can hide instance-level misses. Metrics Reloaded
  recommends choosing metrics by problem type and reporting object-level measures where
  instances matter (Maier-Hein et al., 2024).
- **[F]** SmallFlood-CD's protocol (`object_boundary_patch_v2`):
  - 8-connected components, one-to-one matching at IoU 0.1.
  - Train-fitted small threshold of 35 px.
  - Recall reported separately for *interior* small components (not touching the patch
    edge or invalid pixels) and *clipped* small components.
  - Counts are pooled before scores, globally and per event.
- **[G]** Within these searches, no SAR flood paper was found that separates interior from
  patch-clipped small components. This censoring problem comes from patch tiling and
  affects any patch-based evaluation of small objects.

## 5. Theme 4: size- and instance-aware losses

- **[E]** Tversky loss weights false negatives and false positives asymmetrically to trade
  precision against recall under class imbalance (Salehi et al., 2017).
- **[E]** Dice-style losses handle class imbalance but not *instance* imbalance within a
  class, so large instances dominate small ones. Two losses target this:
  - Blob loss is a per-component loss. Its authors report F1 gains on five 3D medical tasks
    (Kofler et al., 2022, preprint).
  - ICI loss adds instance-wise and centre-of-instance terms and reports DSC gains over
    Dice and blob loss on stroke lesions (Rachmadi et al., 2024).
- **[F]** SmallFlood-CD tested this family on SAR flood change detection under frozen,
  paired controls:
  - Capped inverse-size weights inside BCE and Tversky.
  - The Candidate R local-component loss, which is related to blob/ICI. The project already
    records this prior-art overlap.
  - Results:
    - Size weighting lowered best interior-small recall (B−A = −13.24 pp).
    - Candidate R failed 4 of 5 frozen criteria. Its +4.59 pp interior-small gain came
      with a 33.57 pp drop in component precision and roughly 10× more false-positive
      pixels at the last checkpoint.
- **[G]** Within these searches, every reported gain for instance-aware losses comes from
  medical segmentation with mostly single-image inputs. No study was found that tests these
  losses on SAR change detection, or reports a *controlled negative* transfer result.
- **[H]** Speckle, label generation that removes very small floods, and event shift may
  stop the medical-imaging gains from transferring. This is a hypothesis for discussion
  only. The project's logs do not establish any mechanism, and nothing here authorizes a
  test.

## 6. Theme 5: boundary-aware methods

- **[E]** Boundary loss replaces regional integrals with a contour-distance term to
  stabilize training on highly unbalanced segmentation (Kervadec et al., 2019).
- **[E]** In optical change detection, boundary terms and edge modules are common.
  LRNet, for example, localizes change regions first and then refines them with an
  edge-area alignment module (Zhong et al., 2024, preprint).
- **[F]** The project's boundary package combines a radius-1 morphological boundary target,
  a light residual head and BCE plus Dice. In the 2×2 factorial its best-checkpoint
  boundary-IoU signals did not persist at the last checkpoint. The package also bundles
  head parameters, residual inference and auxiliary supervision, so it cannot isolate their
  separate effects.
- **[G]** Boundary-aware change-detection papers found here are optical and high-resolution
  (building change). Within these searches, no evidence was found for 20 m SAR flood
  boundaries, where the boundary itself is speckle-limited and only a few pixels wide.

## 7. Theme 6: lightweight change-detection architectures

- **[E]** MobileNetV3-Small was designed through hardware-aware architecture search for
  low-resource mobile CPUs (Howard et al., 2019).
- **[E]** Lightweight optical change-detection models report parameters and FLOPs:
  - TinyCD: 13–140× smaller than compared models, Siamese U-Net with low-level feature
    mixing (Codegoni et al., 2022, preprint).
  - LCD-Net: MobileNetV2 encoder, 2.56M parameters, 4.45 GFLOPs (Liu et al., 2024,
    preprint).
- **[F]** SmallFlood-CDNet uses a MobileNetV3-Small Siamese encoder, a hard 5M-parameter
  cap and about 1.08M parameters.
- **[G]** The lightweight change-detection literature found here is evaluated almost
  entirely on optical benchmarks (LEVIR-CD, WHU-CD, SYSU, S2Looking). Lightweight claims in
  SAR flood papers seen at abstract level do not report a consistent cost set. This is
  bounded to abstracts read.

## 8. Theme 7: edge deployment, and what to claim without a Jetson

- **[E]** Onboard flood segmentation has been shown on a low-power accelerator aboard
  PhiSat-1. Transmitting masks instead of images cuts downlink by about two orders of
  magnitude (Mateo-Garcia et al., 2019, 2021). That work uses **optical** imagery.
- **[E]** A full onboard change-detection pipeline (compression, co-registration, change
  detection) sustained 0.7 Mpixel/s on a 15 W accelerator (Inzerillo et al., 2025). That
  work also uses optical imagery.
- **[E]** Parameter count and FLOPs are poor proxies for latency, because they ignore
  parallelism and memory-access cost. Papers should report several cost indicators and say
  which hardware they used (Dehghani et al., 2022).
- **[F]** The frozen deployment YAML targets a Jetson Orin Nano 8GB with TensorRT FP16:
  latency ≤30 ms, peak memory ≤2048 MB, model size ≤20 MB. An ONNX export scaffold exists,
  but there is no TensorRT, parity or Orin evidence. **The user has confirmed that no
  Jetson is available.**
- **[G]** No SAR flood change-detection paper was found that reports on-device latency
  together with accuracy after TensorRT/FP16 conversion. This holds for small-region
  metrics as well as pixel metrics. The onboard examples found are optical.

### Recommended argument for direction (b) without a Jetson [H, needs approval]

Use an **"edge-oriented efficiency, deployment deferred"** framing. Do not use a "Jetson
deployment" framing.

**Claims the paper can make:**
1. Accuracy and efficiency of 0.5M–3.5M-parameter SAR change detectors under one
   controlled protocol.
2. A full cost set, following Dehghani et al. (2022):
   - parameters;
   - MACs/FLOPs at 2×[1,2,256,256];
   - serialized size in FP32 and FP16;
   - peak activation memory;
   - measured latency and throughput on the stated server GPU (RTX A6000), in both
     PyTorch FP32 and TensorRT FP16, with warmup and repetition protocol reported.
3. FP16 conversion fidelity: pixel agreement plus *all* project metrics, including
   interior-small recall and boundary metrics, before and after ONNX/TensorRT FP16. This
   holds for the GPU used, not for Orin.
4. A statement that the models fit the *nominal* frozen Orin budgets on model size (a
   hardware-independent quantity), while latency and memory compliance on Orin remain
   **unverified**.

**Claims the paper must not make:**
- "Real-time on Jetson."
- Any Orin latency extrapolated from A6000 numbers.
- "Deployable onboard," without a stated pipeline. Inputs here are preprocessed dB
  intensity, which onboard use would also require.

**Optional second hardware point.** Measuring CPU latency on a commodity CPU (server CPU
or laptop) would add a second, low-power-relevant point without new hardware. This would
be a new measurement and needs approval.

**Title and wording.** Replace "on Jetson Orin Nano" with "for edge-oriented deployment",
and list Orin measurement as future work. The frozen deployment YAML should be left
unchanged and recorded as an *unmet, deferred* target. Do not edit it to match what is
available.

## 9. Theme 8: synthesis of the research gap

The gap that motivated the frozen design, restated with the evidence now in hand:

1. **[E→G]** Urban flooding is the hard case in SAR. UrbanSARFloods' own FU F1 is below
   0.1 under weighted cross-entropy. Yet the SAR flood change-detection literature found
   here reports pixel metrics, mostly on open-area-dominated benchmarks.
2. **[E→G]** Instance-aware losses help small medical lesions. No controlled test was found
   for SAR flood change detection.
3. **[E→G]** Simple Siamese baselines are competitive in optical change detection. No such
   test was found for SAR urban floods with small-region metrics.
4. **[E→G]** Efficiency claims in remote-sensing change detection often rely on parameters
   and FLOPs, which are known to mislead. No SAR flood paper found reports conversion
   fidelity.

**[F]** What SmallFlood-CD already holds against this gap:
- A frozen, paired 2×2 factorial plus a gated candidate (R).
- An interior/clipped small-region and boundary protocol.
- Pilot evidence that the size-aware and local-component losses did **not** help.

The original positive claim remains unsupported. Direction (b) therefore reframes the
contribution as **controlled evidence plus an evaluation protocol plus an honest efficiency
benchmark**, not as a new method.

**[H] Hypotheses and proposals raised by this review. None is adopted; each needs explicit
approval.**

- H1: The negative loss results hold across seeds. This needs a multi-seed protocol for at
  least A and D.
- H2: FU-pixel recall is much lower than FO recall for every model. This can be measured
  from the retained semantic arrays without retraining, but it is a new analysis.
- H3: The FC-Siam-Diff ≥ heavier-model ordering survives a tuning-fair comparison and a
  single held-out test evaluation.
- H4: FP16 conversion leaves pixel metrics unchanged but shifts small-region or boundary
  metrics more. This is plausible in principle; no SAR-specific source was verified.

## 10. Annotated bibliography (APA 7; identity verified on 2026-10-03)

- Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). Sen1Floods11: A
  georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.
  *Proceedings of the IEEE/CVF CVPR Workshops*.
  https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html
  — Baseline SAR flood dataset with 11 events; deep models outperform thresholding. Theme 1.
- Bountos, N. I., et al. (2024). Kuro Siwo: 33 billion m² under the water. A global
  multi-temporal satellite dataset for rapid flood mapping. *Advances in Neural Information
  Processing Systems, 37* (Datasets and Benchmarks Track).
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/43612b0662cb6a4986edf859fd6ebafe-Abstract.html
  — 43 events, GRD and SLC, BlackBench baselines. Possible external dataset for future
  generalization work. *Full author list to be completed from the proceedings page.* Theme 1.
- Cao, Z., Wu, M., Yan, R., Zhang, F., & Wan, X. (2020). Detection of small changed
  regions in remote sensing imagery using convolutional neural network. *IOP Conference
  Series: Earth and Environmental Science, 502*(1), 012017.
  https://doi.org/10.1088/1755-1315/502/1/012017 — Si-HRNet plus volume-weighted Dice for
  small changes. Theme 3.
- Chen, H., Qi, Z., & Shi, Z. (2022). Remote sensing image change detection with
  transformers. *IEEE Transactions on Geoscience and Remote Sensing, 60*.
  https://arxiv.org/abs/2103.00208 — BIT; the source of the project's BIT-SAR v2 adaptation.
  Theme 2.
- Codegoni, A., Lombardi, G., & Ferrari, A. (2022). *TINYCD: A (not so) deep learning model
  for change detection* [Preprint]. arXiv. https://arxiv.org/abs/2207.13159 — Tiny Siamese
  U-Net with strong optical results. Theme 6.
- Corley, I., Robinson, C., & Ortiz, A. (2024). *A change detection reality check*
  [Workshop paper, ICLR 2024 ML4RS]. arXiv. https://arxiv.org/abs/2402.06994 — A plain U-Net
  remains a top performer; supports controlled-baseline framing. Theme 2.
- Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully convolutional Siamese networks for
  change detection. *IEEE International Conference on Image Processing (ICIP)*.
  https://arxiv.org/abs/1810.08462 — FC-Siam-diff, the project's strongest pilot baseline.
  Theme 2.
- Dehghani, M., Arnab, A., Beyer, L., Vaswani, A., & Tay, Y. (2022). The efficiency
  misnomer. *International Conference on Learning Representations*.
  https://arxiv.org/abs/2110.12894 — Parameters and FLOPs do not predict latency; report
  several indicators. Central to the no-Jetson framing. Theme 7.
- Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y.,
  Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3.
  *IEEE/CVF International Conference on Computer Vision*. https://arxiv.org/abs/1905.02244
  — The project's encoder. Theme 6.
- Inzerillo, G., Valsesia, D., Fiengo, A., & Magli, E. (2025). *Compress-Align-Detect:
  Onboard change detection from unregistered images* [Preprint; reported as published in
  IEEE TGRS]. arXiv. https://arxiv.org/abs/2507.15578 — Onboard change-detection pipeline
  on a 15 W accelerator (optical). Theme 7.
- Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J., & Ben Ayed, I.
  (2019). Boundary loss for highly unbalanced segmentation. *Proceedings of Machine Learning
  Research, 102* (MIDL). https://proceedings.mlr.press/v102/kervadec19a.html — Theme 5.
- Kofler, F., et al. (2022). *blob loss: Instance imbalance aware loss functions for
  semantic segmentation* [Preprint]. arXiv. https://arxiv.org/abs/2205.08209 — Per-component
  loss; prior art for Candidate R. Theme 4.
- Li, Y., Martinis, S., & Wieland, M. (2019). Urban flood mapping with an active
  self-learning convolutional neural network based on TerraSAR-X intensity and
  interferometric coherence. *ISPRS Journal of Photogrammetry and Remote Sensing, 152*,
  178–191. https://elib.dlr.de/127744 — Coherence matters for urban floods; relevant to the
  project's intensity-only limitation. Theme 1.
- Liu, W., Li, J., Wang, H., Tan, R., Fu, Y., & Tian, Q. (2024). *LCD-Net: A lightweight
  remote sensing change detection network combining feature fusion and gating mechanism*
  [Preprint]. arXiv. https://arxiv.org/abs/2410.11580 — Theme 6.
- Maier-Hein, L., Reinke, A., et al. (2024). Metrics reloaded: Recommendations for image
  analysis validation. *Nature Methods, 21*(2), 195–212.
  https://doi.org/10.1038/s41592-023-02151-z — Justifies object-level metrics. Theme 3.
- Mateo-Garcia, G., Oprea, S., Smith, L., Veitch-Michaelis, J., Schumann, G., Gal, Y.,
  Baydin, A. G., & Backes, D. (2019). *Flood detection on low cost orbital hardware*
  [Workshop paper, NeurIPS 2019 AI for HADR]. arXiv. https://arxiv.org/abs/1910.03019 —
  Theme 7.
- Mateo-Garcia, G., Veitch-Michaelis, J., Smith, L., et al. (2021). Towards global flood
  mapping onboard low cost satellites with machine learning. *Scientific Reports, 11*.
  https://doi.org/10.1038/s41598-021-86650-z — PhiSat-1 onboard flood masks (optical).
  Theme 7.
- Rachmadi, M. F., Poon, C., & Skibbe, H. (2024). Improving segmentation of objects with
  varying sizes in biomedical images using instance-wise and center-of-instance segmentation
  loss function. *Proceedings of Machine Learning Research, 227*, 286–300 (MIDL 2023).
  https://proceedings.mlr.press/v227/rachmadi24a.html — ICI loss; prior art for Candidate R.
  Theme 4.
- Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image
  segmentation using 3D fully convolutional deep networks. *Machine Learning in Medical
  Imaging (MLMI)*. https://arxiv.org/abs/1706.05721 — The project uses Tversky with FP 0.3
  and FN 0.7. Theme 4.
- Saleh, T., Weng, X., Holail, S., Hao, C., & Xia, G.-S. (2024a). DAM-Net: Flood detection
  from SAR imagery using differential attention metric-based vision transformers. *ISPRS
  Journal of Photogrammetry and Remote Sensing*. https://arxiv.org/abs/2306.00704 —
  S1GFloods; Siamese SAR flood change detection. Theme 2.
- Saleh, T., Holail, S., Xiao, X., & Xia, G.-S. (2024b). High-precision flood detection and
  mapping via multi-temporal SAR change analysis with semantic token-based transformer.
  *International Journal of Applied Earth Observation and Geoinformation, 131*, 103991.
  https://doi.org/10.1016/j.jag.2024.103991 — Theme 2.
- Tomanič, T., Baudhuin, A., Sotošek, J., Brence, J., Panov, P., Simidjievski, N., & Kocev,
  D. (2026). *A comprehensive and trustworthy benchmark of AI methods for change detection
  in Earth observation* [Preprint]. arXiv. https://arxiv.org/abs/2608.28247 — Siamese U-Nets
  competitive once efficiency is counted; dataset modalities not checked. Theme 2/6.
- Zhao, J., et al. (2024). UrbanSARFloods: Sentinel-1 SLC-based benchmark dataset for urban
  and open-area flood mapping. *Proceedings of the IEEE/CVF CVPR Workshops (EarthVision)*.
  https://arxiv.org/abs/2406.04111 — The project's dataset; FU F1 below 0.1 for the authors'
  baselines. *Full author list to be completed.* Theme 1.
- Zhong, H., Wu, C., & Xiao, Z. (2024). *LRNet: Change detection of high-resolution remote
  sensing imagery via strategy of localization-then-refinement* [Preprint]. arXiv.
  https://arxiv.org/abs/2404.04884 — Theme 5.

### Pending verification (not used as evidence)

| Item | Status |
|---|---|
| "SAR-based flood extent mapping with a lightweight Siamese U-Net and differential attention mechanism", *Earth* (MDPI) 7(3), 87, https://doi.org/10.3390/earth7030087 | DOI resolves; authors, year and cost metrics not retrieved (publisher returned 403). Directly relevant: a lightweight SAR flood change detector on S1GFloods. |
| "High-precision flood change detection with lightweight SAR transformer network and context-aware attention…", ISPRS JPRS (ScienceDirect S0924271625004502) | Title seen in search; authors and metrics not retrieved. |
| INT8/FP16 effects on small objects in segmentation (arXiv 2609.02219, search snippet) | Not opened; claim not used. |

## 11. Limitations of this review

- **Bounded search.** One web engine, 31 queries, no database Boolean search and no
  systematic citation chaining. Absence statements ([G]) mean "not found in these searches".
- **Mostly abstract-level reading.** Reported numbers come from abstracts or landing pages,
  except UrbanSARFloods, which was read at section level.
- **Possibly incomplete recent SAR work.** 2025–2026 lightweight SAR flood papers are
  likely under-covered; two candidates are pending above.
- **Not a novelty clearance.** Before writing the paper, run a targeted database search for
  "UrbanSARFloods" citations and for "SAR flood change detection object-level / small".

## 12. AI disclosure

This review was produced with AI assistance (Claude, via the academic-research-skills
deep-research lit-review workflow). Searches and source checks were run by the AI; every
bibliographic identity listed above was checked against a landing or index page on
2026-10-03. The author must read the key sources in full before citing them in a
manuscript.
