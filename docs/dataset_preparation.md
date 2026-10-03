# UrbanSARFloods preparation on the Ubuntu server

Script: `scripts/prepare_urbansarfloods.py`. Run from the server project
`/data/fit.lamnv/smallflood_cd` in its own `.venv`. Raw source:
`/data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1`.

## Run

```bash
cd /data/fit.lamnv/smallflood_cd
source .venv/bin/activate
python -m pip install numpy rasterio scipy PyYAML
python scripts/prepare_urbansarfloods.py --raw-root /data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1 --plan-only
mkdir -p artifacts/data_preparation
set -o pipefail
python -u scripts/prepare_urbansarfloods.py --raw-root /data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1 2>&1 | tee artifacts/data_preparation/prepare_v1.log
```

Use tmux for the full run. Never run two preparation processes against the same
output directory. `--plan-only` checks pairing, shapes, CRS, transforms and channel
counts and prints the event partition and estimated NPY storage without writing data.
The full run streams images on CPU; no GPU is required. An existing preparation
with identical input file sizes/mtimes, parameters and script can be resumed by
rerunning the exact command. Completed source scenes are reused; a partially
converted scene is rewritten within this script's own output directory. A force
kill may leave `_preparation.lock`; inspect its PID and confirm no preparation
process remains before clearing that lock. Do not remove the raw data or cache.

## Data contract

- Primary target: label 0 remains nonflood; original FO=1 and FU=2 become flood=1.
- Original semantic labels are saved separately; invalid/padded pixels have value
  255 in those semantic arrays and zero in the binary target and valid mask.
- One-based band selection defaults to pre [5,6], post [7,8], interpreted as VH/VV
  intensity already in dB. No second logarithm is applied. These assignments are
  explicit configuration, recorded in metadata, and can be overridden. The TIFF
  samples did not contain band descriptions: the temporal/polarization convention
  still needs independent source verification before final scientific experiments.
- All spectral values used in a patch must be finite and unmasked by the raster.
  Declared GT NoData is excluded; any other valid label outside 0/1/2 is an error.
  Undeclared zero-valued margins cannot automatically be recognized as NoData.
- No arbitrary clipping, per-image contrast normalization or image resizing.
- Four original test frames remain test. The two Jubba tracks share one event.
  Hebei tracks are also grouped. Of remaining flood events, ceil(20%) form
  validation, selected by sorted SHA-256(seed:event), seed 42. Others form train.
  This changes the original train/validation partition. It is event-held-out, not
  necessarily location-held-out: repeated events in one city can cross train/val.
- All full and partial windows are retained, except windows with no valid pixels.
  Partial 256x256 windows are padded and have a valid mask. Keep row/column offsets,
  valid height/width and source geotransforms for reconstruction of test scenes.
- Population mean/std are fitted to valid TRAIN pixels only. Pre and post use the
  same mean/std per polarization. Normalization is already applied to saved NPYs:
  the current `load_array` loader must not normalize them a second time.
- 8-connected components use the exact v1/c8 cache format of the project. Reference
  area is the median TRAIN component area. Small threshold is floor of its 25th
  percentile (minimum 1 pixel), an explicit implementation choice frozen before
  model experiments. Areas are measured within patches and may be clipped by tile
  boundaries. They are not scene-level object areas or physical square metres.

## Outputs (all under the server project)

```text
data/processed/urbansarfloods/
  patches/<source>/..._{pre,post,label,valid_mask,semantic_label}.npy
  component_cache/<hash-prefix>/<hash>.npz
  patch_manifest.csv
  PREPARATION_COMPLETE.json
  _preparation_state.json
  _normalization.json
  _sources/<source>.json
data/metadata/
  normalization_v1.json
  component_stats_v1.json
  preparation_report_v1.json
data/splits/split_v1.yaml
```

Manifest uses absolute server paths. Changing machines or moving the project
requires regenerating/relocating those paths. Header/shape checks cover every
generated NPY before the completion marker is written. The source inventory uses
size/mtime to detect changes for resume, not per-TIFF cryptographic checksums; the
raw archive was separately SHA-256-verified during extraction.

## Remaining scientific gates

This prepares the binary intensity-only experiment. `coherence_path` and
`uncertain_mask_path` are intentionally empty: no coherence feature definition or
author-provided annotation-uncertainty mask has been verified. Do not interpret
with/without-uncertainty runs as distinct experiments on this manifest. Enabling
the coherence model requires an appropriate additional manifest and feature.

Strict matching of CRS/transform/shape verifies a shared geospatial grid; it does
not prove absence of subpixel physical misregistration. Boundary sensitivity tests
and visual quality inspection are still needed. The current experiment evaluator
averages patch metrics; scene reconstruction and scene-level component evaluation
are needed before presenting official benchmark comparisons or full-object recall.
The dataset supports flood extent, not building/road structural damage labels.

## Source references

- Official dataset: https://github.com/jie666-6/UrbanSARFloods
- Author clarification on finalized archive and full-frame testing:
  https://github.com/jie666-6/UrbanSARFloods/issues/3#issuecomment-3043980269
