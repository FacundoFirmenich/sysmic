"""
Geometry Utilities for Sysmic Framework
=======================================

Provides strict geodesic-to-metric transformations to maintain 
isotropy in fractal analysis.
"""

import numpy as np
import math

def geodesic_to_metric(lon, lat, depth_km):
    """
    Convert lon/lat/depth to local metric coordinates (km).
    Depth is preserved (z = depth).
    
    Parameters
    ----------
    lon : array-like
        Longitude in degrees.
    lat : array-like
        Latitude in degrees.
    depth_km : array-like
        Depth in km.
        
    Returns
    -------
    coords_metric : ndarray (N, 3)
        [x, y, z] in km, locally scaled.
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    depth = np.asarray(depth_km, float)

    mean_lat_rad = np.deg2rad(lat.mean())
    km_per_deg_lat = 111.1
    km_per_deg_lon = 111.1 * math.cos(mean_lat_rad)

    x = (lon - lon.min()) * km_per_deg_lon
    y = (lat - lat.min()) * km_per_deg_lat
    z = depth.copy()
    
    return np.column_stack([x, y, z])
