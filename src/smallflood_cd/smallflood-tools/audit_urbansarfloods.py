from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.windows import Window

root = Path("/data/fit.lamnv/UrbanSARFloods/UrbanSARFloods_v1")
train = root / "extracted_v1/urban_sar_floods"
pairs = []
errors = []

# Ghép các chip huấn luyện theo tên, không theo thứ tự danh sách.
for category in ("01_NF", "02_FO", "03_FU"):
    sar_files = sorted((train / category / "SAR").glob("*_SAR.tif"))
    gt_files = set((train / category / "GT").glob("*.tif"))
    matched = set()
    print(category, "SAR:", len(sar_files), "GT:", len(gt_files), flush=True)

    for sar in sar_files:
        gt = train / category / "GT" / sar.name.replace("_SAR.tif", "_GT.tif")
        if not gt.exists():
            errors.append(f"Missing GT: {sar.name}")
            continue
        matched.add(gt)
        event = sar.stem.split("_ID_")[0]
        pairs.append(("development", event, sar, gt))

    for gt in sorted(gt_files - matched):
        errors.append(f"GT without SAR: {gt}")

# Test sử dụng ảnh gốc có nhãn.
for folder in sorted((root / "testing_case_orig").iterdir()):
    if not folder.is_dir():
        continue
    sar = sorted(folder.glob("*_SAR.tif"))
    gt = sorted(folder.glob("*_GT.tif"))
    if len(sar) != 1 or len(gt) != 1:
        errors.append(f"Ambiguous test pair: {folder.name}")
    else:
        pairs.append(("test", folder.name, sar[0], gt[0]))

stats = defaultdict(lambda: np.zeros(5, dtype=np.int64))

for index, (source, event, sar_path, gt_path) in enumerate(pairs, 1):
    try:
        with rasterio.open(sar_path) as sar, rasterio.open(gt_path) as gt:
            if not (
                sar.count == 8 and gt.count == 1
                and sar.crs is not None and sar.crs == gt.crs
                and sar.shape == gt.shape
                and sar.transform.almost_equals(gt.transform)
            ):
                errors.append(f"Grid/band mismatch: {sar_path}")
                continue

            counts = stats[(source, event)]
            counts[0] += 1
            for row in range(0, gt.height, 512):
                for col in range(0, gt.width, 512):
                    window = Window(
                        col, row,
                        min(512, gt.width - col),
                        min(512, gt.height - row),
                    )
                    labels = gt.read(1, window=window)
                    for value in (0, 1, 2):
                        counts[value + 1] += np.count_nonzero(labels == value)
                    counts[4] += np.count_nonzero(
                        ~np.isin(labels, [0, 1, 2])
                    )
    except Exception as exc:
        errors.append(f"{sar_path}: {exc}")

    if index % 500 == 0:
        print(f"Checked {index}/{len(pairs)} pairs", flush=True)

print("\nSOURCE | EVENT | PAIRS | LABEL_0 | LABEL_1 | LABEL_2 | OTHER")
for (source, event), counts in sorted(stats.items()):
    print(source, event, *counts.tolist(), sep=" | ")

print("\nERROR COUNT:", len(errors))
for error in errors[:30]:
    print(error)

development = {event for source, event in stats if source == "development"}
test = {event for source, event in stats if source == "test"}
print("\nEXACT EVENT NAMES SHARED WITH TEST:", sorted(development & test))
print("AUDIT FINISHED")
