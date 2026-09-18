"""Fetch VED and pull out the pure-EV telemetry.

Downloads ~176 MB and expands to ~3 GB of CSV, then keeps only the three pure EVs
(VehId 10, 455, 541 — all 2013 Nissan Leaf) as ev_raw.csv.

    python download_data.py
"""
import glob
import os
import urllib.request

import pandas as pd
import py7zr

BASE = 'https://raw.githubusercontent.com/gsoh/VED/master/Data/'
PARTS = ['VED_DynamicData_Part1.7z', 'VED_DynamicData_Part2.7z']
EV_IDS = {10, 455, 541}
COLS = ['DayNum', 'VehId', 'Trip', 'Timestamp(ms)', 'Latitude[deg]', 'Longitude[deg]',
        'Vehicle Speed[km/h]', 'OAT[DegC]', 'Air Conditioning Power[Watts]',
        'Heater Power[Watts]', 'HV Battery Current[A]', 'HV Battery SOC[%]',
        'HV Battery Voltage[V]']


def main():
    os.makedirs('raw', exist_ok=True)
    for part in PARTS:
        if not os.path.exists(part):
            print('downloading', part)
            urllib.request.urlretrieve(BASE + part, part)
        with py7zr.SevenZipFile(part) as z:
            z.extractall('raw')

    frames = []
    for f in sorted(glob.glob('raw/*.csv')):
        for chunk in pd.read_csv(f, usecols=COLS, chunksize=500_000, low_memory=False):
            hit = chunk[chunk.VehId.isin(EV_IDS)]
            if len(hit):
                frames.append(hit)

    df = pd.concat(frames, ignore_index=True).sort_values(['VehId', 'Trip', 'Timestamp(ms)'])
    df.to_csv('ev_raw.csv', index=False)
    print(f'{len(df):,} rows, {df.groupby(["VehId", "Trip"]).ngroups} trips -> ev_raw.csv')


if __name__ == '__main__':
    main()
