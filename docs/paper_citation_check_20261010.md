# Citation check for the ICEIR 2026 draft

## Material Passport

Scope: the 24 references of `paper/iceir2026/refs.bib` and the sentences that cite them in
`main.tex`, checked on 2026-10-10 in a Claude Code session with web search and page fetches.
This is an identity and claim-alignment check at **abstract or landing-page level**. No full
text was read except where stated in the 2026-10-03 literature review. Policy: advisory
(mark only), as recorded in the Paper Configuration Record.

## 1. Identity and bibliographic data

| Reference | Checked on | Result |
|---|---|---|
| Zhao, Xiong, Zhu 2024 (UrbanSARFloods) | 10-10, arXiv page | Authors and CVPR 2024 EarthVision workshop confirmed |
| Bonafilia et al. 2020 (Sen1Floods11) | 10-10, search result of the CVF page | Authors confirmed; sources disagree on pages, so none are given |
| Bountos et al. 2024 (Kuro Siwo) | 10-10, NeurIPS proceedings page | Nine authors added; Datasets and Benchmarks Track confirmed |
| Saleh et al. 2024a (DAM-Net) | 10-10, search | Volume 212, pages 440–453 taken from the first author's publication list; **publisher page not retrieved** |
| Kaçmaz and Alganci 2026 | 10-10, institutional record | Authors, journal, volume, issue, article number and DOI confirmed |
| Daudt et al. 2018; Chen et al. 2022; Corley et al. 2024; Dehghani et al. 2022 | 10-10, arXiv pages | Authors confirmed; Chen et al. DOI 10.1109/TGRS.2021.3095166 |
| Tomanič et al. 2026 | 10-10, arXiv page | Seven authors confirmed (given names corrected from the page) |
| Kofler et al. 2022 (blob loss) | 10-10, arXiv page | 17 authors; abbreviated to three plus "et al." deliberately |
| Maier-Hein et al. 2024 | 10-10, search (publisher page blocked) | Journal, volume, issue, pages, DOI and first three authors confirmed; consortium list abbreviated |
| Mateo-Garcia et al. 2021 | 10-10, search (publisher page blocked) | Eight authors, volume 11, article 7249, DOI confirmed |
| Codegoni et al. 2022 | 10-10, arXiv page | Authors confirmed; preprint only |
| Bouthillier et al. 2021; Picard 2021; Reimers and Gurevych 2017 | 10-10, arXiv / proceedings / ACL Anthology | Confirmed |
| Li et al. 2019; Saleh et al. 2024b; Salehi et al. 2017; Rachmadi et al. 2024; Kervadec et al. 2019; Cao et al. 2020; Howard et al. 2019 | 10-03 only | Identity verified in the literature review; not re-fetched on 10-10 |

## 2. Claim alignment (sentence versus source)

| Sentence in the draft | Finding | Action |
|---|---|---|
| "Evaluation guidelines recommend object-level metrics when objects of very different sizes matter" (Maier-Hein et al.) | Too specific for what the abstract states (problem-driven metric selection covering object detection and instance segmentation) | Reworded to problem-driven metric choice, including object-level validation |
| "Instance-aware losses weight each connected component rather than each pixel" (Kofler; Rachmadi) | The blob-loss abstract describes the aim (instance-level detection, small instances overlooked by Dice), not the mechanism | Reworded to the stated aim |
| "interferometric coherence helps in built-up areas" (Li et al.) | Only the title and abstract-level reading support "intensity and coherence were combined" | Reworded to "combined intensity with coherence"; same in Limitations |
| "Siamese designs with attention have been applied to SAR flood pairs" (Saleh 2024a, 2024b) | Second paper is a token-based transformer | Reworded to "Siamese and transformer-based designs" |
| "flood segmentation has been developed for use on board small satellites" (Mateo-Garcia et al.) | Source is optical imagery on a satellite with an accelerator | Reworded to "from optical imagery ... demonstrated for on-board processing on a small satellite" |
| Sen1Floods11 "11 flood events"; Kuro Siwo "43 events", GRD and SLC; UrbanSARFloods "8,879 chips of 512×512 from 18 events", urban detection "remains challenging" | Match the abstracts | None |
| Corley et al. (plain U-Net remains a top performer); Tomanič et al. (Siamese U-Nets competitive once cost is counted); Dehghani et al. (cost indicators can disagree); Codegoni et al. (small model, strong accuracy); Kaçmaz and Alganci (lightweight SAR flood model) | Match the abstracts | None |
| Reimers and Gurevych; Bouthillier et al.; Picard (seed and variance effects) | Match the abstracts | None |

## 3. Limits of this check

- Abstract or landing-page level only; a claim that depends on the body of a paper was not
  verified. The draft uses these sources for general statements, not for numbers.
- One bibliographic detail (DAM-Net volume and pages) comes from the author's own list, not
  the publisher.
- Seven references were not re-fetched on 2026-10-10.
- The gap statement in the draft remains a statement about bounded searches.
