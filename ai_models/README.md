# ai_models/

All machine learning models live here, fully separated from the web
application. The backend consumes models only through the interfaces in
[`base.py`](base.py) — never by importing training code.

## Layout

| Path | Purpose | Phase |
|---|---|---|
| `base.py` | Abstract `RainfallModel` interface + `ModelOutput` contract | 1 (done) |
| `baseline/` | Tabular classifier on meteorological features (scikit-learn); forecasts the **next day** per point | 3 (done) |
| `satellite_model/` | CNN on NASA GIBS MODIS imagery (PyTorch); classifies the **same day** per region | 4 (done) |
| `fusion_model/` | Hybrid engine combining both branches into the final risk probability, confidence and historical analogues | 5 (done) |
| `saved_models/` | Trained artifacts (git-ignored; distribute via releases or DVC) | 3+ |
| `experiments/` | One JSON record per training run — metrics, selection reason, artifacts | 3+ |
| `tabular/`, `vision/`, `hybrid/` | Empty namespace stubs kept from the Phase 1 skeleton | — |

Each model package follows the same shape: `config.py`, data/dataset
preparation, `model.py` or `fusion.py`, `train.py`, `evaluate.py`,
`predict.py`.

## Rules

- Training scripts, experiment configs, and evaluation reports belong in the
  respective model package; exploratory work goes in `notebooks/`.
- Every trained artifact must carry a version and a metrics report before it
  is served.
- Models must populate `ModelOutput.explanation_context` so the
  `explainability/` package can compute SHAP values and Grad-CAM heatmaps
  without reaching into model internals.
- A feature derived from another model's output must be generated
  **out-of-fold**. The Phase 3 artifact had memorised most of the Phase 5
  period (AUC 1.000 in-sample vs 0.875 out-of-sample), which made every
  fusion approach score a meaningless validation macro-F1 of 1.000. See
  `fusion_model/dataset.py` and docs/hybrid_ai_architecture.md.
- Reports state what the numbers do *not* show. Every model report ends
  with a CAVEATS block regenerated from measured values — unmeasurable
  classes, underpowered splits, and baselines the model failed to beat
  are named explicitly.
