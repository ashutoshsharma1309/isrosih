# VARUNA AI — Prediction Pipeline

End-to-end flow from observations to a warning. Code:
`ai_models/fusion_model/predict.py` (`HybridPredictor`).

## 1. Complete flow

```
                        Data Input
        weather observations · satellite scene · location · timestamp
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
     Weather Model (Phase 3)          Satellite Model (Phase 4)
   ai_models/baseline/…pkl          ai_models/satellite_model/…pt
   tabular → class probs            scene → class probs
   weather_risk_score               satellite_risk_score
              │                                │
              │                    scene statistics (cloud density,
              │                    cold-top fraction, growth, dispersion)
              └───────────────┬────────────────┘
                              ▼
                      Fusion Engine (Phase 5)
              feature assembly → selected fusion strategy
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
     class probabilities  historical      confidence
                          analogues       estimation
                              │
                              ▼
                       Final Prediction
        event_prediction · risk_probability · confidence · risk_level
```

## 2. Input

The predictor drives the upstream models itself, so a caller supplies
only what it observed:

```json
{
  "weather_data": {
    "temperature_c": 25.8, "humidity_pct": 90.9, "pressure_hpa": 1000.0,
    "wind_speed_ms": 6.4, "wind_direction_deg": 255.0, "cloud_cover_pct": 97.0,
    "rain_sum_1d": 103.1, "rain_sum_3d": 180.1,
    "latitude": 10.0, "longitude": 76.3
  },
  "satellite_image_path": "data/raw/satellite/.../kerala_2018-08-15.jpg",
  "ir_image_path": "data/raw/satellite/.../kerala_2018-08-15.jpg",
  "location": {"region": "Kerala", "latitude": 10.0, "longitude": 76.3},
  "timestamp": "2018-08-15T05:15:00Z"
}
```

Alternatives when a branch cannot be run locally: pass
`satellite_features` with `class_probabilities`, or a precomputed
`weather_risk_score` / `satellite_risk_score`. A scalar score cannot say
how risk splits between heavy and extreme, so the reconstruction assigns
it all to heavy and records that in `notes`.

## 3. Processing steps

1. **Validate** — types, latitude/longitude ranges, ISO timestamp,
   image existence. Malformed input raises `InvalidInputError`; nothing
   is silently coerced.
2. **Weather branch** — the Phase 3 bundle scores the supplied
   conditions, producing a class-probability vector and
   `weather_risk_score`.
3. **Satellite branch** — the Phase 4 checkpoint classifies the scene;
   `extract_scene_features` computes the interpretable statistics from
   the True Color (and, when given, IR and previous) images.
4. **Feature assembly** — the fusion vector is built. Season terms are
   derived from the timestamp. Anything still missing falls back to the
   training median and is listed in `imputed_features` rather than being
   invented silently.
5. **Fusion** — the selected strategy returns class probabilities;
   `risk_probability` = P(heavy) + P(extreme).
6. **Historical analogues** — the three most similar past high-impact
   events, excluding the query day itself.
7. **Confidence** — prediction probability, branch agreement and
   historical similarity, blended per `docs/hybrid_ai_architecture.md`.

## 4. Output

```json
{
  "event_prediction": "Heavy Rainfall",
  "risk_probability": 0.5272,
  "confidence": "Medium",
  "confidence_pct": 61.6,
  "risk_level": "HIGH",
  "predicted_category": 1,
  "class_probabilities": {"Normal": 0.4728, "Heavy": 0.5272, "Extreme": 0.0},
  "contributing_models": {
    "weather_risk_score": 0.5272,
    "satellite_risk_score": 0.8376,
    "agreement": 0.6896
  },
  "similar_historical_events": [
    {"event": "Mumbai 2019-08-02 (Heavy, 66 mm)", "similarity_pct": 73.0}
  ],
  "imputed_features": ["cloud_growth_rate"],
  "model": "weighted_fusion_w1.00-v1"
}
```

Real output for Kerala, 15 Aug 2018 — a day whose observed next-day
outcome was Heavy.

`risk_level` thresholds `risk_probability`: LOW ≤ 0.25, MODERATE ≤ 0.50,
HIGH ≤ 0.75, CRITICAL above. `confidence` is High ≥ 75%, Medium ≥ 50%,
else Low — the same vocabulary as the Phase 3 and Phase 4 predictors.

## 5. Running it

```bash
backend/.venv/bin/python -m ai_models.fusion_model.predict --input '{...}'
```

Exit code 1 with `{"error": "..."}` on stderr when input is invalid or no
fusion model has been trained.

## 6. Failure modes

| Condition | Behaviour |
|---|---|
| No trained fusion bundle | `ModelNotTrainedError` naming the train command |
| Neither branch evaluable | `InvalidInputError` — no prediction is fabricated |
| Image unreadable | scene features dropped, prediction continues, warning logged |
| Missing feature | training median substituted, listed in `imputed_features` |
| Historical matcher unavailable | empty analogue list; confidence renormalises |

## 7. Not yet wired

The backend still returns 501 for predictions — connecting
`HybridRainfallModel` to `PredictionService` is Phase 7, and the
explanation payloads (SHAP, Grad-CAM) are Phase 6. `explanation_context`
already carries the raw material both will need.
