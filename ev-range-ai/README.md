# EV range predictor

Predicts energy use (Wh/km) and remaining range for an electric car from how it is
being driven, trained on real OBD-II telemetry rather than synthetic data.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:5000
```

The trained model ships in `model/`, so the app runs without re-downloading anything.

## Rebuild from raw data

```bash
python download_data.py      # ~176 MB from the VED repo -> ev_raw.csv
python prepare_dataset.py    # 10 Hz logs -> ev_windows.csv (180 s windows)
python train_model.py        # -> model/ev_model.pkl + model/model_meta.json
```

## Data

[VED (Vehicle Energy Dataset)](https://github.com/gsoh/VED), University of Michigan —
383 cars logged over OBD-II in Ann Arbor, Nov 2017 to Nov 2018. Only the three pure
EVs are used (all 2013 Nissan Leaf, 24 kWh pack): 476,308 readings at 10 Hz across
504 trips.

Each trip is cut into 180-second windows. Per window:

| Feature | Meaning |
|---|---|
| `avg_speed`, `max_speed`, `speed_std` | speed profile, km/h |
| `avg_accel` | mean positive acceleration, m/s² |
| `brake_events_per_km` | samples below −1 m/s², per km |
| `idle_frac` | share of time below 1 km/h |
| `oat` | outside air temperature, °C |
| `aux_power_w` | air conditioning + heater draw |
| `soc` | battery state of charge at window start |

Target `wh_per_km` is integrated from HV battery voltage × current over the window,
divided by distance integrated from speed. Windows outside 30–600 Wh/km are dropped as
sensor dropouts. Result: 1,420 windows over 469 trips.

## Model

Random forest, chosen over histogram gradient boosting on cross-validated error.

| | This model | Always guess the median |
|---|---|---|
| Wh/km | MAE 44.2, R² 0.56 | MAE 63.2 |
| Range | MAE 37.0 km | MAE 60.4 km |

Validation is `GroupKFold` by trip. Windows from one trip are highly correlated, so a
random row split leaks trip identity and reports a flattering score that does not hold
on a new drive.

**Range is derived, not predicted.** Remaining range is exactly
`usable energy left / Wh per km`, so a second regression head can only re-learn that
identity with extra noise. Measured both ways under the same CV: learned head 48.7 km
MAE, derived 37.0 km MAE.

Feature importance: auxiliary power 0.27, idle fraction 0.23, average speed 0.16,
outside temperature 0.08. Climate load and stop-and-go dominate — expected for a small
pack in Michigan winters.

## Coaching

`/api/predict` re-runs the model with one input improved at a time (climate off, no
idling, gentler acceleration, less braking, 60 km/h cap) and reports the difference. The
advice is the model's own counterfactual, so it can never contradict the number above it.

## API

`POST /api/predict` — all nine features required.

```json
{"avg_speed": 45, "max_speed": 65, "speed_std": 14, "avg_accel": 0.5,
 "brake_events_per_km": 5, "idle_frac": 0.08, "oat": 12,
 "aux_power_w": 400, "soc": 80}
```

```json
{"wh_per_km": 131.5, "range_km": 131.4, "energy_left_kwh": 17.28,
 "percentile": 37, "fleet_median": 153.7,
 "tips": [{"label": "Accelerate gently", "wh_saved": 18.8,
           "pct_saved": 14.3, "extra_km": 21.9}]}
```

Missing or non-numeric inputs return `400` with an `error` field.

`GET /api/meta` — features, valid ranges, accuracy, feature importance, fleet deciles.

## Limitations

- One car model (2013 Nissan Leaf) in one city. Numbers will not transfer to a modern
  long-range EV without retraining.
- No road grade: VED has GPS but no elevation, and slope matters for energy use.
- R² 0.56 means real spread remains. Traffic and driver behaviour inside a window are
  only partly captured by nine aggregates.
- Battery degradation is not modelled; usable capacity is fixed at 90% of 24 kWh.
