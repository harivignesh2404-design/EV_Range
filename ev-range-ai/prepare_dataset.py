"""Turn raw VED EV telemetry into a windowed training table.

Input : ev_raw.csv   (10 Hz OBD-II logs for the 3 pure EVs in VED, 2013 Nissan Leaf)
Output: ev_windows.csv  (one row per 60 s driving window)

Targets
  wh_per_km : energy drawn from the HV battery per km over the window
  range_km  : remaining range = usable energy left in pack / wh_per_km
"""
import numpy as np
import pandas as pd

WINDOW_S = 180.0
PACK_KWH = 24.0        # 2013 Nissan Leaf advertised capacity
USABLE_FRAC = 0.90     # usable share of the pack
BRAKE_THRESHOLD = -1.0  # m/s^2, counts as a braking event

COLS = {
    'Vehicle Speed[km/h]': 'speed',
    'HV Battery Current[A]': 'current',
    'HV Battery Voltage[V]': 'voltage',
    'HV Battery SOC[%]': 'soc',
    'OAT[DegC]': 'oat',
    'Air Conditioning Power[Watts]': 'ac_w',
    'Heater Power[Watts]': 'heater_w',
    'Timestamp(ms)': 'ts',
}


def load():
    df = pd.read_csv('ev_raw.csv').rename(columns=COLS)
    df = df.dropna(subset=['speed', 'current', 'voltage', 'soc'])
    df[['ac_w', 'heater_w']] = df[['ac_w', 'heater_w']].fillna(0.0)
    df['oat'] = df['oat'].fillna(df['oat'].median())
    return df.sort_values(['VehId', 'Trip', 'ts'])


def window_features(w):
    dt = w['dt'].to_numpy()
    speed = w['speed'].to_numpy()
    mps = speed / 3.6

    dist_km = float((mps * dt).sum() / 1000.0)
    # Discharge is negative current in VED, so flip the sign for energy drawn.
    wh = float((-w['voltage'].to_numpy() * w['current'].to_numpy() * dt).sum() / 3600.0)
    if dist_km < 0.05 or wh <= 0:
        return None

    accel = np.gradient(mps, np.cumsum(dt))
    accel = np.clip(accel, -8, 8)
    moving = speed > 1.0

    soc_start = float(w['soc'].iloc[0])
    wh_per_km = wh / dist_km
    usable_wh_left = PACK_KWH * 1000 * USABLE_FRAC * soc_start / 100.0

    return {
        'avg_speed': float(speed[moving].mean()) if moving.any() else 0.0,
        'max_speed': float(speed.max()),
        'speed_std': float(speed.std()),
        'avg_accel': float(accel[accel > 0].mean()) if (accel > 0).any() else 0.0,
        'brake_events_per_km': float((accel < BRAKE_THRESHOLD).sum() * dt.mean() / dist_km),
        'idle_frac': float((~moving).mean()),
        'oat': float(w['oat'].mean()),
        'aux_power_w': float((w['ac_w'] + w['heater_w']).mean()),
        'soc': soc_start,
        'wh_per_km': wh_per_km,
        'range_km': usable_wh_left / wh_per_km,
        'trip_key': f"{int(w['VehId'].iloc[0])}-{int(w['Trip'].iloc[0])}",
    }


def build(df):
    rows = []
    for _, trip in df.groupby(['VehId', 'Trip'], sort=False):
        trip = trip.copy()
        trip['dt'] = trip['ts'].diff().fillna(100.0).clip(0, 2000) / 1000.0
        trip['bucket'] = ((trip['ts'] - trip['ts'].iloc[0]) / 1000.0 // WINDOW_S).astype(int)
        for _, w in trip.groupby('bucket'):
            if w['dt'].sum() < WINDOW_S * 0.5:
                continue
            f = window_features(w)
            if f:
                rows.append(f)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    out = build(load())
    # Drop physically implausible windows (sensor dropouts, regen-dominated coasting).
    out = out[out['wh_per_km'].between(30, 600)]
    out.to_csv('ev_windows.csv', index=False)
    print(out.shape, out['trip_key'].nunique(), 'trips')
    print(out.drop(columns='trip_key').describe().round(2).to_string())
