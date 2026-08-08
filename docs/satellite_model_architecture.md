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

### Candidates compared

| Name | Architecture | Trainable params | Rationale |
|---|---|---|---|
| `custom_cnn` | 4 conv blocks (32→64→128→256, BN+ReLU+MaxPool) → GAP → dropout → linear | ~1.1 M | From-scratch baseline; no pretraining assumptions |
| `resnet18` | ImageNet-pretrained; layer4 + fc fine-tuned, rest frozen | ~8.4 M | Transfer learning — pretrained texture/edge features transfer well to cloud imagery |
| `vit_b_16` | ImageNet-pretrained ViT-B/16, frozen backbone, linear head | ~2.3 K | The computationally honest ViT option (linear probe) per the "if feasible" spec clause |

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
