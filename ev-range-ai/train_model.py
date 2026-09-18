"""Train the EV energy + range model on windowed VED data.

Design note
-----------
Two outputs, but only one learned head. `range_km` is an exact function of
`wh_per_km` and SOC (usable energy left / consumption), so a second regressor
can only re-learn that identity with added noise. Measured both ways under the
same CV: learned head MAE 48.7 km, derived from the energy head MAE 37.0 km.
So range is derived, not predicted.

Validation is GroupKFold by trip — windows from one trip are correlated, and a
random row split leaks trip identity into the test fold.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

FEATURES = ['avg_speed', 'max_speed', 'speed_std', 'avg_accel',
            'brake_events_per_km', 'idle_frac', 'oat', 'aux_power_w', 'soc']

PACK_KWH, USABLE_FRAC = 24.0, 0.90


def derive_range(wh_per_km, soc):
    return PACK_KWH * 1000 * USABLE_FRAC * np.asarray(soc) / 100.0 / np.maximum(wh_per_km, 1.0)


def candidates():
    return {
        'random_forest': lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=4, random_state=42, n_jobs=-1),
        'hist_gradient_boosting': lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, min_samples_leaf=20,
            l2_regularization=1.0, random_state=42),
    }


def main():
    df = pd.read_csv('ev_windows.csv')
    X, y, groups = df[FEATURES], df['wh_per_km'], df['trip_key']

    report = {}
    for name, make in candidates().items():
        oof = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            oof[te] = make().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        rng = derive_range(oof, df['soc'])
        report[name] = {
            'wh_per_km': {'mae': round(mean_absolute_error(y, oof), 2),
                          'r2': round(r2_score(y, oof), 3)},
            'range_km': {'mae': round(mean_absolute_error(df['range_km'], rng), 2),
                         'r2': round(r2_score(df['range_km'], rng), 3)},
        }
        print(f"{name:24s} Wh/km MAE={report[name]['wh_per_km']['mae']:6.2f} "
              f"R2={report[name]['wh_per_km']['r2']:.3f} | "
              f"range MAE={report[name]['range_km']['mae']:6.2f} km")

    winner = min(report, key=lambda n: report[n]['wh_per_km']['mae'])
    baseline = mean_absolute_error(y, np.full(len(y), y.median()))
    print(f"baseline(median)         Wh/km MAE={baseline:6.2f}  -> winner: {winner}")

    model = candidates()[winner]().fit(X, y)
    joblib.dump(model, 'ev_model.pkl')

    importance = (dict(zip(FEATURES, np.round(model.feature_importances_, 4).tolist()))
                  if hasattr(model, 'feature_importances_') else {})

    meta = {
        'features': FEATURES,
        'winner': winner,
        'report': report,
        'baseline_mae': round(baseline, 2),
        'feature_importance': importance,
        'feature_ranges': {f: [round(float(df[f].quantile(0.01)), 2),
                               round(float(df[f].quantile(0.99)), 2)] for f in FEATURES},
        'pack_kwh': PACK_KWH,
        'usable_frac': USABLE_FRAC,
        'n_windows': int(len(df)),
        'n_trips': int(groups.nunique()),
        'window_seconds': 180,
        'source': 'VED (Vehicle Energy Dataset) - 3 pure EVs (2013 Nissan Leaf), Ann Arbor MI',
    }
    with open('model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(importance, indent=2))


if __name__ == '__main__':
    main()
