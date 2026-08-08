# explainability/

The Explainable AI layer (Phase 6). Every prediction VARUNA AI serves must
be able to say why it was made — numerically, spatially, and in plain
language.

## Layout

| Path | Purpose |
|---|---|
| `base.py` | `Explainer` contract fixed in Phase 1 |
| `shap_explainer/` | SHAP attribution over the fusion pipeline (`config`, `explainer`, `visualization`) |
| `gradcam_explainer/` | Captum Grad-CAM over the Phase 4 CNN (`config`, `gradcam`, `visualization`) |
| `explanation_generator/` | Unified payload and natural-language narrative (`generator`, `templates`) |
| `historical_explainer.py` | Nearest past analogues, wrapping the Phase 5 matcher |
| `reports/` | `writer.py` — CLI that renders reports and figures |

Outputs go to `reports/shap/`, `reports/gradcam/`, and
`reports/xai_report_v1.txt` / `reports/xai_explanations_v1.json`.

```bash
backend/.venv/bin/python -m explainability.reports.writer
backend/.venv/bin/python -m explainability.reports.writer --region Kerala --date 2018-08-15
```

## Rules

- **Attribute the model that actually reads the input.** The Phase 5
  engine is a pipeline, not one estimator. Measured: perturbing humidity
  changes the shipped `WeightedFusion` output by exactly 0.000000, because
  it reads only the upstream probability vectors. SHAP therefore runs on
  the branch that consumes each input and is scaled by that branch's
  fusion weight — the chain rule, exact for a linear blend.
- **Never invent an attribution.** Inputs the selected model provably
  cannot read are listed in `unattributed_inputs` with the reason. A
  branch on zero weight reports exactly zero.
- **Attributions must reconstruct the prediction.** `base_value + Σ
  contributions == predicted_risk`, enforced by test.
- **Prose states only what was observed.** Median-filled inputs and
  geometry/calendar terms stay in the attribution table but out of the
  narrative — and how much attribution they absorbed is disclosed.
- **Heatmaps come from gradients, never from noise.** Grad-CAM uses the
  target layer Phase 4 recorded in the checkpoint, upsampled bilinearly
  for display only.
- **Carry the caveats.** `caveats` is a first-class output field. An
  explanation that hides its limits invites more trust than the evidence
  supports.

See `docs/explainable_ai_architecture.md` for the workflows and
`docs/ai_explanation_examples.md` for real output.
