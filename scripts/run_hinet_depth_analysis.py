"""
CRITICAL TEMPORAL - DEPTH 896 KM
All temporal analyses with extended depth (0-896 km) for deep subduction
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import io

from sfa.core import FractalDimensionEstimator
from sfa.stats import SeismicityAnalysis

END_DATE = datetime(2025, 12, 12)

EVENTS = {
    'Nankai_Trough': {'lat': 33.0, 'lon': 135.0, 'magnitude': None, 'regime': 'Subduction'},
    'Noto_M7.6': {'lat': 37.49, 'lon': 137.27, 'magnitude': 7.6, 'regime': 'Strike-slip intraplate'}
}

TEMPORAL_WINDOWS_DAYS = [24, 48, 96, 128, 192, 312, 365, 730, 1095, 1460]
RADIUS_KM = 192
MIN_DEPTH = 0
MAX_DEPTH = 896  # EXTENDED FOR DEEP SUBDUCTION
MIN_MAG = 0.8
MAX_MAG = 9.6
BOOTSTRAP = 200
MIN_EVENTS = 6
USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def fetch_catalog(lat, lon, start_date, end_date, timeout=180):
    params = {
        'format': 'csv', 'latitude': lat, 'longitude': lon, 'maxradiuskm': RADIUS_KM,
        'starttime': start_date.strftime('%Y-%m-%d'), 'endtime': end_date.strftime('%Y-%m-%d'),
        'minmagnitude': MIN_MAG, 'maxmagnitude': MAX_MAG,
        'mindepth': MIN_DEPTH, 'maxdepth': MAX_DEPTH, 'orderby': 'time'
    }
    try:
        response = requests.get(USGS_API, params=params, timeout=timeout)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        if df.empty:
            return pd.DataFrame(columns=['time', 'latitude', 'longitude', 'depth', 'mag'])
        keep = ['time', 'latitude', 'longitude', 'depth', 'mag']
        available = [c for c in keep if c in df.columns]
        return df[available].dropna()
    except Exception as e:
        return None

def compute_d2_d3(df):
    if df is None or len(df) < MIN_EVENTS:
        return {'n_events': len(df) if df is not None else 0, 'd2': np.nan, 'd2_sem': np.nan,
                'd3_est': np.nan, 'b_value': np.nan, 'b_unc': np.nan, 'mc': np.nan}
    
    lats, lons = df['latitude'].values, df['longitude'].values
    depths = df['depth'].values if 'depth' in df.columns else np.zeros(len(lats))
    mean_lat = np.mean(lats)
    x_km = (lons - np.mean(lons)) * 111.1 * np.cos(np.radians(mean_lat))
    y_km = (lats - np.mean(lats)) * 111.1
    z_km = depths
    
    coords = np.column_stack([x_km, y_km, z_km])
    ranges = coords.max(axis=0) - coords.min(axis=0)
    ranges[ranges == 0] = 1.0
    coords_norm = (coords - coords.min(axis=0)) / ranges
    
    try:
        estimator = FractalDimensionEstimator()
        d2, d2_sem = estimator.compute_gp_dimension(coords_norm, bootstrap_iterations=BOOTSTRAP, return_diagnostics=False)
    except:
        d2, d2_sem = np.nan, np.nan
    
    try:
        seismicity = SeismicityAnalysis()
        b_val, b_unc, mc = seismicity.compute_b_value(df['mag'].dropna().values)
    except:
        b_val, b_unc, mc = np.nan, np.nan, np.nan
    
    d3_estimate = d2 / 0.75 if not np.isnan(d2) else np.nan
    return {'n_events': len(df), 'd2': d2, 'd2_sem': d2_sem, 'd3_est': d3_estimate,
            'b_value': b_val, 'b_unc': b_unc, 'mc': mc}

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*80)
    print("  CRITICAL TEMPORAL - DEPTH 0-896 KM")
    print("="*80)
    print(f"  MAX_DEPTH: {MAX_DEPTH} km (deep subduction)")
    print(f"  Windows: {TEMPORAL_WINDOWS_DAYS}")
    print(f"  Timestamp: {timestamp}")
    print("="*80)
    
    results_all = []
    total = len(EVENTS) * len(TEMPORAL_WINDOWS_DAYS)
    current = 0
    
    for event_name, event_config in EVENTS.items():
        print(f"\n{'='*80}\nEVENT: {event_name}\n{'='*80}")
        
        for window_days in TEMPORAL_WINDOWS_DAYS:
            current += 1
            start_date = END_DATE - timedelta(days=window_days)
            print(f"\n[{current}/{total}] Window = {window_days} days")
            print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
            print(f"  Fetching catalog...")
            
            df = fetch_catalog(event_config['lat'], event_config['lon'], start_date, END_DATE)
            
            if df is not None:
                n_events = len(df)
                rate_per_day = n_events / window_days
                print(f"  Retrieved: {n_events} events ({rate_per_day:.3f} ev/day)")
            else:
                n_events, rate_per_day = 0, 0.0
                print(f"  ERROR fetching")
            
            if df is not None and len(df) >= MIN_EVENTS:
                print(f"  Computing D2/D3 (bootstrap n={BOOTSTRAP})...")
                results = compute_d2_d3(df)
                print(f"    D₂ = {results['d2']:.3f} ± {results['d2_sem']:.3f}")
                print(f"    D₃ = {results['d3_est']:.3f}")
                print(f"    b-value = {results['b_value']:.3f} ± {results['b_unc']:.3f}")
                print(f"    Mc = {results['mc']:.2f}")
            else:
                print(f"  Insufficient events ({n_events} < {MIN_EVENTS})")
                results = compute_d2_d3(df)
            
            row = {'timestamp': timestamp, 'event': event_name, 'magnitude': event_config.get('magnitude'),
                   'regime': event_config['regime'], 'lat': event_config['lat'], 'lon': event_config['lon'],
                   'radius_km': RADIUS_KM, 'min_depth': MIN_DEPTH, 'max_depth': MAX_DEPTH,
                   'min_mag': MIN_MAG, 'window_days': window_days,
                   'start_date': start_date.strftime('%Y-%m-%d'), 'end_date': END_DATE.strftime('%Y-%m-%d'),
                   'rate_per_day': rate_per_day, **results}
            results_all.append(row)
    
    df_results = pd.DataFrame(results_all)
    output_csv = Path(__file__).parent / f"hinet_temporal_depth896_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(output_csv, index=False)
    
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE (DEPTH 0-896 KM)")
    print("="*80)
    print(f"  Output: {output_csv}")
    print("="*80)
    
    print("\n" + "="*80)
    print("  N(t) SUMMARY (DEPTH 0-896 KM)")
    print("="*80)
    for event_name in EVENTS.keys():
        df_event = df_results[df_results['event'] == event_name]
        print(f"\n{event_name}:")
        for _, row in df_event.iterrows():
            print(f"  {int(row['window_days']):4d}d: N={int(row['n_events']):4d} ({row['rate_per_day']:.4f} ev/d)")

if __name__ == "__main__":
    main()
