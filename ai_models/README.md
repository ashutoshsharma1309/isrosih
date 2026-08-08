# ai_models/

All machine learning models live here, fully separated from the web
application. The backend consumes models only through the interfaces in
[`base.py`](base.py) — never by importing training code.

## Layout

| Path | Purpose | Phase |
|---|---|---|
| `base.py` | Abstract `RainfallModel` interface + `ModelOutput` contract | 1 (done) |
| `tabular/` | Baseline classifier on meteorological features (gradient boosting / random forest, scikit-learn) | 3 |
| `vision/` | CNN on INSAT/NASA satellite imagery (PyTorch) — cloud density, formation and movement patterns | 4 |
| `hybrid/` | Fusion model combining tabular + vision signals into the final risk probability | 5 |
| `artifacts/` | Trained model weights (git-ignored; tracked via releases or DVC) | 3+ |

## Rules

- Training scripts, experiment configs, and evaluation reports belong in the
  respective model package; exploratory work goes in `notebooks/`.
- Every trained artifact must carry a version and a metrics report before it
  is served.
- Models must populate `ModelOutput.explanation_context` so the
  `explainability/` package can compute SHAP values and Grad-CAM heatmaps
  without reaching into model internals.
