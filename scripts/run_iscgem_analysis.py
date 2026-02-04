"""
ISC-GEM M8+ TEMPORAL ANALYSIS (PRE-MAIN-POST)
Analyzes all M8+ earthquakes 1964-2021 with temporal windows
Uses ISC-GEM catalog for background seismicity
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sysmic.core import FractalDimensionEstimator
from sysmic.stats import SeismicityAnalysis
from sysmic.system import Sysmic
from sysmic.infrastructure import SystemConfiguration, CertificationLevel

# Configuration
ISCGEM_CATALOG = Path(__file__).parent / "isc-gem-cat.csv"
M8_LIST = Path(__file__).parent / "m8plus_1964_2021_list.csv"
RADIUS_KM = 192  # Standard
MAX_DEPTH = 896  # Deep events allowed
MIN_EVENTS = 10
BOOTSTRAP = 200

# Temporal windows (days before mainshock)
PRE_WINDOWS = [24, 48, 64, 96, 128, 192]
MAIN_DAYS = 16  # Mainshock window (±8 days)
POST_WINDOWS = [16, 32, 64]

def load_iscgem():
    """Load ISC-GEM catalog."""
    print("Loading ISC-GEM catalog...")
    # ISC-GEM CSV has no header row; data starts at line 118 (after 117 comment lines)
    # Column names from line 117 comment
    col_names = ['date', 'lat', 'lon', 'smajax', 'sminax', 'strike', 'q', 'depth', 
                 'unc', 'q2', 'mw', 'unc2', 'q3', 's', 'mo', 'fac', 'mo_auth', 
                 'mpp', 'mpr', 'mrr', 'mrt', 'mtp', 'mtt', 'str1', 'dip1', 'rake1',
                 'str2', 'dip2', 'rake2', 'type', 'eventid']
    df = pd.read_csv(ISCGEM_CATALOG, skiprows=117, names=col_names, 
                     skipinitialspace=True, low_memory=False)
    df['time'] = pd.to_datetime(df['date'])
    df['mag'] = df['mw']
    print(f"  Loaded: {len(df):,} events (1904-2021)")
    return df

def load_m8_events():
    """Load M8+ event list."""
    print("Loading M8+ events...")
    df = pd.read_csv(M8_LIST)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  M8+ events: {len(df)}")
    return df

def filter_spatial(df_catalog, center_lat, center_lon, radius_km, max_depth):
    """Spatial filter around epicenter."""
    mean_lat = center_lat
    dx = (df_catalog['lon'] - center_lon) * 111.1 * np.cos(np.radians(mean_lat))
    dy = (df_catalog['lat'] - center_lat) * 111.1
    dist = np.sqrt(dx**2 + dy**2)
    
    mask = (dist <= radius_km) & (df_catalog['depth'] <= max_depth)
    return df_catalog[mask].copy()

def compute_d2_d3(df):
    """Compute D2/D3 and b-value."""
    if len(df) < MIN_EVENTS:
        return {
            'n_events': len(df), 'd2': np.nan, 'd2_sem': np.nan,
            'd3_est': np.nan, 'b_value': np.nan, 'b_unc': np.nan, 'mc': np.nan
        }
    
    lats, lons, depths = df['lat'].values, df['lon'].values, df['depth'].values
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
        d2, d2_sem = estimator.compute_gp_dimension(coords_norm, bootstrap_iterations=BOOTSTRAP)
    except:
        d2, d2_sem = np.nan, np.nan
    
    try:
        seismicity = SeismicityAnalysis()
        mags = df['mag'].dropna().values
        if len(mags) >= MIN_EVENTS:
            b_val, b_unc, mc = seismicity.compute_b_value(mags)
        else:
            b_val, b_unc, mc = np.nan, np.nan, np.nan
    except:
        b_val, b_unc, mc = np.nan, np.nan, np.nan
    
    return {
        'n_events': len(df),
        'd2': d2,
        'd2_sem': d2_sem,
        'd3_est': d2/0.75 if not np.isnan(d2) else np.nan,
        'b_value': b_val,
        'b_unc': b_unc,
        'mc': mc
    }

def analyze_m8_event(event, df_catalog):
    """Analyze one M8+ event with PRE-MAIN-POST windows."""
    eq_date = event['date']
    eq_lat = event['lat']
    eq_lon = event['lon']
    eq_mag = event['mw']
    
    print(f"\n{'='*70}")
    print(f"  {eq_date.date()} M{eq_mag:.1f} ({eq_lat:.2f}, {eq_lon:.2f})")
    print(f"{'='*70}")
    
    # Spatial filter
    df_spatial = filter_spatial(df_catalog, eq_lat, eq_lon, RADIUS_KM, MAX_DEPTH)
    print(f"  Spatial filter (R={RADIUS_KM}km): {len(df_spatial):,} events available")
    
    results = []
    
    # PRE windows
    for pre_days in PRE_WINDOWS:
        start = eq_date - timedelta(days=pre_days)
        end = eq_date - timedelta(days=1)
        
        df_window = df_spatial[(df_spatial['time'] >= start) & (df_spatial['time'] < end)]
        
        print(f"\n  PRE_{pre_days}d: {start.date()} to {end.date()}")
        print(f"    N = {len(df_window)}")
        
        if len(df_window) >= MIN_EVENTS:
            res = compute_d2_d3(df_window)
            print(f"    D₂ = {res['d2']:.3f} ± {res['d2_sem']:.3f}, D₃ = {res['d3_est']:.3f}")
            
            results.append({
                'event_date': eq_date,
                'event_lat': eq_lat,
                'event_lon': eq_lon,
                'event_mag': eq_mag,
                'window_type': 'PRE',
                'window_days': pre_days,
                'start_date': start,
                'end_date': end,
                **res
            })
        else:
            print(f"    Insufficient events (N<{MIN_EVENTS})")
    
    # MAIN window
    main_start = eq_date - timedelta(days=MAIN_DAYS//2)
    main_end = eq_date + timedelta(days=MAIN_DAYS//2)
    df_main = df_spatial[(df_spatial['time'] >= main_start) & (df_spatial['time'] <= main_end)]
    
    print(f"\n  MAIN_{MAIN_DAYS}d: {main_start.date()} to {main_end.date()}")
    print(f"    N = {len(df_main)}")
    
    if len(df_main) >= MIN_EVENTS:
        res = compute_d2_d3(df_main)
        print(f"    D₂ = {res['d2']:.3f} ± {res['d2_sem']:.3f}, D₃ = {res['d3_est']:.3f}")
        
        results.append({
            'event_date': eq_date,
            'event_lat': eq_lat,
            'event_lon': eq_lon,
            'event_mag': eq_mag,
            'window_type': 'MAIN',
            'window_days': MAIN_DAYS,
            'start_date': main_start,
            'end_date': main_end,
            **res
        })
    
    # POST windows
    for post_days in POST_WINDOWS:
        start = eq_date + timedelta(days=1)
        end = eq_date + timedelta(days=post_days)
        
        df_window = df_spatial[(df_spatial['time'] > start) & (df_spatial['time'] <= end)]
        
        print(f"\n  POST_{post_days}d: {start.date()} to {end.date()}")
        print(f"    N = {len(df_window)}")
        
        if len(df_window) >= MIN_EVENTS:
            res = compute_d2_d3(df_window)
            print(f"    D₂ = {res['d2']:.3f} ± {res['d2_sem']:.3f}, D₃ = {res['d3_est']:.3f}")
            
            results.append({
                'event_date': eq_date,
                'event_lat': eq_lat,
                'event_lon': eq_lon,
                'event_mag': eq_mag,
                'window_type': 'POST',
                'window_days': post_days,
                'start_date': start,
                'end_date': end,
                **res
            })
    
    return results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*70)
    print("  ISC-GEM M8+ TEMPORAL ANALYSIS")
    print("="*70)
    print(f"  Period: 1964-2021")
    print(f"  Radius: {RADIUS_KM} km")
    print(f"  Max depth: {MAX_DEPTH} km")
    print(f"  Bootstrap: {BOOTSTRAP}")
    print("="*70)
    
    # Load data
    df_catalog = load_iscgem()
    df_m8 = load_m8_events()
    
    print(f"\nAnalyzing {len(df_m8)} M8+ events...")
    
    all_results = []
    
    for idx, event in df_m8.iterrows():
        try:
            results = analyze_m8_event(event, df_catalog)
            all_results.extend(results)
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            continue
    
    # Save results
    df_results = pd.DataFrame(all_results)
    output_csv = Path(__file__).parent / f"iscgem_m8plus_temporal_{timestamp}.csv"
    df_results.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"  COMPLETE")
    print(f"{'='*70}")
    print(f"  Total analyses: {len(all_results)}")
    print(f"  Output: {output_csv.name}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
