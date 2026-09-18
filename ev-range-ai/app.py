"""EV range and energy predictor — Flask API.

Model is trained on VED (Vehicle Energy Dataset) OBD-II logs from three pure EVs.
See train_model.py for validation details.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'model', 'ev_model.pkl')
META_PATH = os.path.join(BASE, 'model', 'model_meta.json')
DATA_PATH = os.path.join(BASE, 'ev_windows.csv')

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    META = json.load(f)
FEATURES = META['features']
PACK_WH = META['pack_kwh'] * 1000 * META['usable_frac']

# Fleet distribution, used to place a drive against the real training windows.
FLEET = np.sort(pd.read_csv(DATA_PATH)['wh_per_km'].to_numpy())

# Each lever is (label, feature, target value given the current one).
LEVERS = [
    ('Turn the climate control down', 'aux_power_w', lambda v: max(0.0, v - 600)),
    ('Stop idling', 'idle_frac', lambda v: min(v, 0.02)),
    ('Accelerate gently', 'avg_accel', lambda v: min(v, 0.30)),
    ('Coast into stops', 'brake_events_per_km', lambda v: min(v, 2.0)),
    ('Hold 60 km/h', 'avg_speed', lambda v: min(v, 60.0)),
]


def predict_wh_per_km(row):
    X = pd.DataFrame([row], columns=FEATURES)
    return float(model.predict(X)[0])


def range_km(wh_per_km, soc):
    return PACK_WH * soc / 100.0 / max(wh_per_km, 1.0)


def coaching(row, base_wh, soc):
    """Re-run the model with one input improved at a time.

    The saving is what the model itself predicts, not a hand-written rule, so the
    advice always matches the model that produced the number above it.
    """
    base_range = range_km(base_wh, soc)
    out = []
    for label, feature, better in LEVERS:
        current = row[feature]
        target = better(current)
        if abs(target - current) < 1e-6:
            continue
        trial = dict(row)
        trial[feature] = target
        if feature == 'avg_speed':
            trial['max_speed'] = max(target, row['max_speed'] - (current - target))
        saved = base_wh - predict_wh_per_km(trial)
        if saved <= 0.5:
            continue
        out.append({
            'label': label,
            'wh_saved': round(saved, 1),
            'pct_saved': round(100 * saved / base_wh, 1),
            'extra_km': round(range_km(base_wh - saved, soc) - base_range, 1),
        })
    return sorted(out, key=lambda t: -t['wh_saved'])[:4]


@app.route('/')
def home():
    return render_template('index.html')


def histogram(n_bins=24):
    counts, edges = np.histogram(FLEET, bins=n_bins)
    return {'edges': [round(float(e), 1) for e in edges],
            'counts': [int(c) for c in counts]}


@app.route('/api/meta')
def meta():
    return jsonify({
        'features': FEATURES,
        'feature_ranges': META['feature_ranges'],
        'importance': META['feature_importance'],
        'accuracy': META['report'][META['winner']],
        'baseline_mae': META['baseline_mae'],
        'model': META['winner'],
        'n_windows': META['n_windows'],
        'n_trips': META['n_trips'],
        'source': META['source'],
        'fleet_deciles': [round(float(np.percentile(FLEET, p)), 1) for p in range(0, 101, 10)],
        'histogram': histogram(),
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(silent=True) or {}
        row = {}
        for f in FEATURES:
            if f not in body:
                return jsonify({'error': f'Missing input: {f}'}), 400
            row[f] = float(body[f])

        if row['max_speed'] < row['avg_speed']:
            row['max_speed'] = row['avg_speed']

        wh = predict_wh_per_km(row)
        soc = row['soc']
        percentile = float(np.searchsorted(FLEET, wh) / len(FLEET) * 100)

        return jsonify({
            'wh_per_km': round(wh, 1),
            'range_km': round(range_km(wh, soc), 1),
            'energy_left_kwh': round(PACK_WH * soc / 100 / 1000, 2),
            'percentile': round(percentile),
            'fleet_median': round(float(np.median(FLEET)), 1),
            'tips': coaching(row, wh, soc),
        })
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400
    except Exception as e:
        app.logger.exception('predict failed')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
