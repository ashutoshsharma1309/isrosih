# VARUNA AI — Satellite Data Processing (Phase 4)

## Imagery source (real, openly accessible)

While MOSDAC INSAT registration is pending, the vision model trains on
**NASA GIBS/Worldview** MODIS Terra scenes (open, no-auth NASA service,
daily coverage since 2000, ~10:30 local overpass):

| Layer | Use |
|---|---|
| `MODIS_Terra_CorrectedReflectance_TrueColor` | Model input — visible cloud structure |
| `MODIS_Terra_Brightness_Temp_Band31_Day` (~11 µm thermal IR) | Feature extraction — cloud-top temperature proxy (cold convective tops appear dark) |

Acquisition: `data_pipeline/collectors/gibs_collector.py` (concurrent,
idempotent, provenance-cataloged; region bboxes padded to ≥4° because
rain-bearing systems are synoptic-scale). Dataset build:
`data_pipeline/build_gibs_imagery.py`.

## Labeling — no manual annotation, no invented labels

Each scene is labeled with the **real rainfall outcome** for its
region-day from the validated Phase 2/3 dataset: label = max IMD
category over the region's grid points that day (0 normal / 1 heavy /
2 extreme). All 485 positive region-days were fetched, plus a normal
sample stratified by region × month (ratio 1.5:1) so the model cannot
cheat by associating a region or season with a class. Result: 1,144
scenes with both layers (2001–2024).

Index: `data/labels/satellite_labels.parquet`
(region, date, true_color_path, ir_path, label).

## Preprocessing (`ai_models/satellite_model/preprocessing.py`)

Training pipeline (per scene):
1. Decode JPEG → RGB (undecodable files raise `ImageLoadError`)
2. Resize → 224 × 224
3. Augmentation (train only): random horizontal flip, ±15° rotation,
   brightness/contrast jitter (±20%)
4. `ToTensor` + ImageNet mean/std normalization (required by the
   pretrained backbones) → float32 tensor (3, 224, 224)

Evaluation pipeline is identical minus augmentation (deterministic —
verified by test).

## Interpretable scene features (for Phase 5 fusion)

`extract_scene_features()` computes physically meaningful statistics per
scene, all masked against **swath gaps** (MODIS black wedges are excluded
via a validity mask; a scene <5% valid raises rather than yielding
garbage):

| Feature | Meaning |
|---|---|
| `cloud_density` | Fraction of valid pixels above the cloud brightness threshold |
| `brightness_mean/std` | Luminance statistics (cloud thickness proxy) |
| `cold_top_fraction` | IR: share of very cold pixels = deep convection (None when IR missing — never invented) |
| `spatial_dispersion` | Evenness of cloud cover over a 4×4 grid (organised system vs scattered) |
| `cloud_growth_rate` | 0.5 + Δcloud_density vs previous scene (None without history) |
| `valid_fraction` | Share of the scene actually imaged |

Batch output for every indexed scene:
`data/processed/features/satellite_features.parquet`.

## Dataset splitting

Chronological by date (70/15/15), like Phase 3 — the test set is a
genuinely unseen future period. Class imbalance handled with
inverse-frequency weights in the loss.

## Known limitations

- MODIS is 1 snapshot/day; the T-3h…T sequence input in the Phase 4 spec
  needs INSAT's 30-minute cadence — the architecture accepts it later
  (temporal stacking documented in satellite_model_architecture.md), and
  `cloud_growth_rate` already captures day-scale motion.
- Swath gaps occasionally cover part of a region; features use validity
  masking, and the CNN sees the gap as black (consistent between train
  and inference).
- 3 extreme-class scenes — same root cause as Phase 3 (label source
  resolution), same planned fix (IMD gridded / IMERG labels).
