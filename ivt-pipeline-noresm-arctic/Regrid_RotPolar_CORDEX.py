#!/usr/bin/env python
# coding: utf-8

import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import os
import sys

def wrap_lon(IVT):
    '''
    Wrap longitude coordinate of an xarray DataArray or Dataset along 'lon'.
    Assumes longitude in 0-360 range, wraps by adding one ghost point before and after.

    '''
    print("🔁 Wrapping first and last longitude to handle global continuity...")

    lon = IVT.lon.values               # get lon values
    left_wrap_lon = lon[-1] - 360      # point before 0 (e.g., ~358.6 - 360 = -1.40625)
    right_wrap_lon = lon[0] + 360      # point after last (e.g., 0 + 360 = 360)

    left_wrap_data = IVT.isel(lon=-1).assign_coords(lon=left_wrap_lon) #add the data 359 to -1
    right_wrap_data = IVT.isel(lon=0).assign_coords(lon=right_wrap_lon) #add the data from 0 tot 360

    wrapped_da = xr.concat([left_wrap_data, IVT, right_wrap_data], dim="lon") #concat to 1 dataset with now having 2 more lon points
    
    print("✅ Longitude wrapping successful.")
    
    return wrapped_da

def add_polar_point(IVT_wrapped):
    '''
    Adds a synthetic North Pole (90° latitude) point to an IVT dataset by averaging 
    the values at the highest existing latitude across all longitudes.

    Raises:
        ValueError: If the dataset already includes a 90° latitude point.
    '''
    if 90.0 in IVT_wrapped.lat:
        raise ValueError("Data already contains a 90° latitude.")
    
    print("➕ Adding synthetic North Pole point (lat=90°) based on average of top latitude...")

    # calculate the average value of the grid points surrounding the pole
    polar_value = IVT_wrapped.sel(lat=IVT_wrapped.lat[-1]).mean(dim='lon')

    # create logistics for lat lon info
    lon_vals = IVT_wrapped.lon.values
    new_lat=90

    # make sure polar_value has the same shape as the IVT data
    polar_expanded = (
        polar_value
        .expand_dims({'lon': lon_vals}, axis=-1)
        .expand_dims('lat')
        .assign_coords(lat=[new_lat])
    )
    # concatenate IVT with polar point
    IVT_with_pole = xr.concat([IVT_wrapped, polar_expanded], dim='lat').sortby('lat')
    
    print("✅ Polar point added successfully.")

    return IVT_with_pole

def regrid_to_ArcticCORDEX(IVT_with_pole, base_dir):
    """
    Regrids a DataArray with lat/lon to RACMO2.4 CORDEX Arctic grid.
    Target grid is hardcoded: 
    """
    print("📐 Regridding to CORDEX Arctic target grid...")

    ds_gcm = IVT_with_pole
    gcm_lat = IVT_with_pole.lat.values
    gcm_lon = IVT_with_pole.lon.values
    gcm_vals = IVT_with_pole.values  # shape: (time, lat, lon)
    
    #load target grid CORDEX RACMO2.4
    target_file = os.path.join(
        base_dir,
        "ivt-pipeline",
        "tas_ARC-12_NorESM2-MM_historical_r1i1p1f1_UU-IMAU_RACMO24P-NN_v1-r1_day_20110101-20141231.nc"
    )    
    ds_target = xr.open_dataset(target_file)
    
    lat_target = ds_target['lat'].values  # shape (rlat, rlon)
    lon_target = ds_target['lon'].values  # shape (rlat, rlon)
    rlat = ds_target['rlat'].values
    rlon = ds_target['rlon'].values
    
    # Convert target lon to 0-360 range if needed
    lon_target_360 = np.where(lon_target < 0, lon_target + 360, lon_target)
    
    # Prepare interpolation points
    points_interp = np.stack([lat_target.ravel(), lon_target_360.ravel()], axis=-1)  # shape (N, 2)
    
    # Interpolate per timestep
    interpolated_data = []
    for t in range(gcm_vals.shape[0]):
        interpolator = RegularGridInterpolator(
            (gcm_lat, gcm_lon),
            gcm_vals[t],
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        interp_t = interpolator(points_interp).reshape(lat_target.shape)
        interpolated_data.append(interp_t)
    
    interpolated_data = np.stack(interpolated_data)  # shape (time, rlat, rlon)
    
    # Create DataArray with metadata
    IVT_regridded = xr.DataArray(
        interpolated_data,
        dims=("time", "rlat", "rlon"),
        coords={
            "time": ds_gcm.time,
            "rlat": rlat,
            "rlon": rlon,
            "lat": (("rlat", "rlon"), lat_target),
            "lon": (("rlat", "rlon"), lon_target)
        },
        name="IVT",
        attrs={
            "units": IVT_with_pole.attrs.get("units", "unknown"),
            "long_name": "Integrated Vapor Transport regridded"
        }
    )

    print("✅ Regridding to 2D lat-lon target grid completed.")
    return IVT_regridded

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python Regrid_RotPolar_CORDEX.py <YEAR> <MONTH> <INPUT_DIR> <OUTPUT_DIR> <BASE_DIR>")
        sys.exit(1)

    year = sys.argv[1]
    month = sys.argv[2]
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]
    base_dir = sys.argv[5]

    input_file = os.path.join(input_dir, f"IVT_CNRM_{year}{month}.nc")
    output_file = os.path.join(output_dir, f"IVT_CNRM_CORDEX_{year}{month}.nc")

    print(f"📂 Loading IVT input: {input_file}")
    IVT_org = xr.open_dataset(input_file)
    IVT = IVT_org.IVT

    IVT_wrapped = wrap_lon(IVT)
    IVT_with_pole = add_polar_point(IVT_wrapped)
    IVT_regridded = regrid_to_ArcticCORDEX(IVT_with_pole,base_dir)

    print(f"💾 Saving regridded IVT to: {output_file}")
    IVT_regridded.to_netcdf(output_file)

    print("✅ Done.")
