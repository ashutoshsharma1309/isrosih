# VARUNA AI — Satellite Model Architecture (Phase 4)

Code: `ai_models/satellite_model/` · Checkpoint:
`ai_models/saved_models/satellite_model_v1.pt` · Report:
`reports/satellite_model_report_v1.txt`

## 1. Approach

Deep-learning scene classification: one satellite cloud scene → the
region-day's rainfall category (0 normal / 1 heavy / 2 extreme) plus a
**satellite risk score** = P(heavy) + P(extreme). Labels are real
observed outcomes (see satellite_data_processing.md) — the model learns
which cloud structures actually co-occur with high-impact rain.

### Candidates implemented

| Name | Architecture | Trainable params | Rationale | Status |
|---|---|---|---|---|
| `custom_cnn` | 4 conv blocks (32→64→128→256, BN+ReLU+MaxPool) → GAP → dropout → linear | ~1.1 M | From-scratch baseline; no pretraining assumptions | **trained & selected** |
| `resnet18` | ImageNet-pretrained; layer4 + fc fine-tuned, rest frozen | ~8.4 M | Transfer learning — pretrained texture/edge features transfer well to cloud imagery | not evaluated |
| `vit_b_16` | ImageNet-pretrained ViT-B/16, frozen backbone, linear head | ~2.3 K | The computationally honest ViT option (linear probe) per the "if feasible" spec clause | not evaluated |

**The shipped v1 checkpoint is `custom_cnn`, selected as the only
candidate that ran — not as the winner of a three-way comparison.** The
report (`reports/satellite_model_report_v1.txt`) shows a single row in its
comparison table for exactly this reason.

Both pretrained candidates are implemented and reachable; they were not
evaluated for environmental reasons, recorded here so the gap is not
mistaken for a result:

1. On the first run, `torchvision` could not download the ImageNet weights
   — `SSL: CERTIFICATE_VERIFY_FAILED`, the usual missing-CA-bundle problem
   on a framework Python install. `train.py` caught it and skipped both
   candidates, leaving `custom_cnn` unopposed. Fix: point `SSL_CERT_FILE`
   and `REQUESTS_CA_BUNDLE` at `certifi.where()`.
2. With the weights cached, training them exhausted the 8 GB development
   machine: epoch time for even the small CNN degraded from 45 s to 300 s
   as the system swapped, and the pretrained backbones were not viable.
   `batch_size_by_model` (32 / 16 / 8) was added to reduce activation
   memory, which is necessary but not sufficient on this hardware.

To complete the comparison on a machine with adequate memory:

```bash
export SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')
backend/.venv/bin/python -m ai_models.satellite_model.train
```

Nothing else needs changing — all three candidates are in `CANDIDATES`,
and selection is already by validation macro-F1.

Training: AdamW, class-weighted cross-entropy (inverse frequency),
augmentation on the train split only, chronological 70/15/15 split,
selection on validation macro-F1, single held-out test evaluation.
Device auto-resolution: Apple MPS > CUDA > CPU.

## 2. Input format

- Single scene: RGB 224×224 tensor, ImageNet-normalized (see
  satellite_data_processing.md for the full pipeline).
- **Sequence readiness:** the spec's T-3h…T-0 sequence input requires
  INSAT's 30-min cadence (MOSDAC). The design accommodates it without
  retraining machinery changes: per-frame CNN features mean-pooled before
  the head, and day-scale motion is already captured as the
  `cloud_growth_rate` feature. Documented as the MOSDAC upgrade path —
  not faked with duplicated frames.

## 3. Output format

`predict.py` (`SatellitePredictor`) returns, per scene:

```json
{
  "satellite_risk_score": 0.87,
  "cloud_pattern": "High Risk",
  "cloud_condition": "Heavy",
  "category": 1,
  "class_probabilities": {"Normal": 0.11, "Heavy": 0.77, "Extreme": 0.12},
  "confidence": "High",
  "scene_features": {"cloud_density": 0.82, "cold_top_fraction": 0.41, "...": "..."},
  "model": "resnet18-v1"
}
```

Vocabulary (risk-score semantics, High/Medium/Low confidence) matches the
Phase 3 tabular predictor exactly, so the Phase 5 fusion consumes both
uniformly. `SatelliteRainfallModel` (model.py) additionally adapts the
checkpoint to the project-wide `RainfallModel` interface.

## 4. Training pipeline

`python -m ai_models.satellite_model.train` produces:
- checkpoint (state dict + architecture + label names + metrics +
  `gradcam_target_layer` — the final conv block, pre-registered for the
  Phase 6 Grad-CAM explainer; ViT records its final norm for attention
  rollout instead)
- VARUNA AI SATELLITE MODEL REPORT (metrics + comparison + selection reason)
- visualizations in `reports/`: training curves, confusion matrix,
  sample prediction grid
- `data/processed/features/satellite_features.parquet` — interpretable
  scene features for every indexed scene (Phase 5 fusion input)
- experiment record in `ai_models/experiments/`

## 5. Evaluation

Macro-F1 selection (accuracy reported with the imbalance caveat), macro
precision/recall, per-class precision/recall/support, confusion matrix.
Real results: see `reports/satellite_model_report_v1.txt` and
docs/baseline_model_results.md's Phase 4 counterpart in the experiment
records.
