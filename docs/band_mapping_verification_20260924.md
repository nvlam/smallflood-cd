# UrbanSARFloods band mapping: source verification

Verified on 2026-09-24 using AI-assisted source retrieval and direct Figure 1 inspection.
Scope: published eight-band dataset convention, not a per-file binary integrity audit.
No training, source-array conversion, or checkpoint modification performed.

## Verdict

The current one-based pre=[5,6], post=[7,8], with channels [VH,VV], agrees with
Figure 1 and explicit public clarifications from repository owner jie666-6.
Earlier project notes saying the public band convention was unverified are superseded by this report.
Those earlier searches had not retrieved the relevant issue replies; the replies date to January 2026.

| One-based band | Feature | Polarization |
|---|---|---|
| 1 | Pre-event coherence | VH |
| 2 | Pre-event coherence | VV |
| 3 | Co-event coherence | VH |
| 4 | Co-event coherence | VV |
| 5 | Pre-event intensity, dB | VH |
| 6 | Pre-event intensity, dB | VV |
| 7 | Post-event intensity, dB | VH |
| 8 | Post-event intensity, dB | VV |

Rasterio uses band indexes 5,6 / 7,8. After read() returns a NumPy band-first array,
equivalent indexes are [4,5] / [6,7]. Do not logarithm-transform the released dB intensities again.
The project metadata reports normalization already applied to prepared NPY files; do not normalize twice.

## Primary evidence

1. Figure 1, visually inspected: all eight numbered bands, VH/VV ordering, intensity color scale in dB.
   https://arxiv.org/html/2406.04111v1/figures/dataset_overview_v2_7.png
   Paper: https://arxiv.org/html/2406.04111v1
2. Repository owner, issue 7, 2026-01-20: Figure 1 is authoritative; section 3 prose is
   an information-type summary rather than an ordering specification. The reference to
   co-event intensity there is a typo for co-event coherence.
   https://github.com/jie666-6/UrbanSARFloods/issues/7#issuecomment-3771836010
3. Repository owner, issue 8, 2026-01-21: confirms log intensity and the explicitly
   proposed band/date pairing. Houston example: coherence 18+24 August and 24+30 August 2017;
   intensity pre=24 August, post=30 August, in bands 5-6 and 7-8 respectively.
   Question: https://github.com/jie666-6/UrbanSARFloods/issues/8#issuecomment-3776417243
   Answer: https://github.com/jie666-6/UrbanSARFloods/issues/8#issuecomment-3777054622
4. Repository owner, issue 10, 2026-01-29: urban_sar_floods.tar.gz is the final release;
   train_validation.rar should not be used; XML sidecars are not processing graphs;
   NaNs are invalid/non-observed SAR regions.
   https://github.com/jie666-6/UrbanSARFloods/issues/10#issuecomment-3816803573

## Retrieval and limits

Read public README, Hugging Face dataset card/discussions, paper HTML and Figure 1.
Web-indexed GitHub pages omitted newer issue/comment content. Read public GitHub REST API
issues?state=all&per_page=100, then comments for issues 7,8,9,10,3; author_association=OWNER
for the confirmations above. No messages or issues were posted.

Evidence grade: direct maintainer clarification plus author-produced paper figure. These sources
are mutually consistent but not independent validations of every distributed raster file.
No separate assertion that every testing_case_orig raster was individually inspected.
The eight-band documented convention is now confirmed; testing-file consistency remains an audit item.

## Additional research caveat

Paper section 3.2.1 says small isolated objects were removed during semi-automatic label creation
with case-dependent thresholds. Section 3.2.2 also discusses removal of small floods unrecognizable
at SAR resolution. Thus small-component performance measures agreement with the released labels,
not completeness for all physical small floods. This limits claims about small-change recovery.

Owner clarification on effective resolution versus pixel spacing:
https://github.com/jie666-6/UrbanSARFloods/issues/9#issuecomment-3771853821
Do not convert a 35-pixel threshold into a fixed square-meter area solely from the paper's 20 m label.

## Implication

No evidence-based reason to reorder existing input channels or regenerate the prepared data.
The public band-order blocker for the next train/validation engineering pilot is resolved.
This does not authorize launching a training job or establish scene-level evaluation readiness.
