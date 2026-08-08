# explainability/

The differentiating layer of VARUNA AI: every prediction ships with an
explanation of *why* the model produced it.

## Planned components (Phase 6)

- `shap_explainer.py` — SHAP `TreeExplainer`/`KernelExplainer` over the
  tabular model's meteorological features. Output: signed contribution per
  feature (humidity, pressure drop, cloud density, ...).
- `gradcam.py` — Grad-CAM over the vision model's final convolutional
  layers. Output: heatmap overlay highlighting the cloud regions that drove
  the prediction, rendered to an image served by the backend.
- `narrative.py` — converts attributions + heatmap statistics into the
  plain-language sentence shown in alerts, e.g. *"Increased cloud density
  and atmospheric moisture contributed to this prediction."*

All implementations conform to [`base.py`](base.py) and emit payloads
matching the API schema in
`backend/app/schemas/prediction.py::Explanation`.
