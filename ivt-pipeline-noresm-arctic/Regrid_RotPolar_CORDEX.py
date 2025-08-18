#!/usr/bin/env python
# coding: utf-8
# NorESM

import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
def wrap_lon(IVT):
    """
    Wrap a 1D lon axis into –180…+180, add  
    one ghost point before and after, then sort.
    """
    import xarray as xr

    print("🔁 Wrapping longitude with ghost points…")

    # 1) remap into –180…+180
    lon_mod = ((IVT.lon + 180) % 360) - 180
    IVT2    = IVT.assign_coords(lon=lon_mod)

    # 2) compute ghost longitudes
    lon_vals     = IVT2.lon.values
    left_ghost   = lon_vals[-1] - 360    # one step before the first real cell
    right_ghost  = lon_vals[0]  + 360    # one step after the last real cell

    # 3) slice out the edge profiles & re-label them
    left_da  = IVT2.isel(lon=-1).assign_coords(lon=left_ghost)
    right_da = IVT2.isel(lon= 0).assign_coords(lon=right_ghost)

    # 4) assemble [left ghost | IVT2 | right ghost], then sort
    wrapped = xr.concat([left_da, IVT2, right_da], dim="lon")
    wrapped = wrapped.sortby("lon")

    print("✅ Longitude wrapping successful:",
          "lon.min/max →", wrapped.lon.min().item(),
                         wrapped.lon.max().item())
    return wrapped


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
def regrid_to_ArcticCORDEX(IVT_wrapped, base_dir):
    """
    Regrid a DataArray (time, lat, lon) on a regular GCM grid
    onto the RACMO2.4 CORDEX Arctic rotated-pole grid.
    Assumes IVT_wrapped.lon & .lat are strictly monotonic.
    """
    print("📐 Regridding to CORDEX Arctic target grid…")

    # 1) Pull out the monotonic coords + data
    gcm_lat  = IVT_wrapped.lat.values      # 1D, strictly increasing
    gcm_lon  = IVT_wrapped.lon.values      # 1D, strictly increasing
    gcm_vals = IVT_wrapped.values          # shape (time, lat, lon)

    # 2) Load the CF-compliant CORDEX example grid
    target_file = os.path.join(base_dir, "ivt-pipeline", "example_grid.nc")
    ds_tgt      = xr.open_dataset(target_file)
    lat_tgt     = ds_tgt["lat"].values     # 2D (rlat, rlon)
    lon_tgt     = ds_tgt["lon"].values     # 2D (rlat, rlon)
    rlat        = ds_tgt["rlat"].values    # 1D
    rlon        = ds_tgt["rlon"].values    # 1D

    # 3) Build the (lat,lon) points to sample
    pts = np.stack([lat_tgt.ravel(), lon_tgt.ravel()], axis=-1)

    # 4) Interpolate each time-slice
    out = []
    for t in range(gcm_vals.shape[0]):
        interp = RegularGridInterpolator(
            (gcm_lat, gcm_lon),
            gcm_vals[t],
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )
        arr = interp(pts).reshape(lat_tgt.shape)
        out.append(arr)
    data = np.stack(out, axis=0)  # (time, rlat, rlon)

    # 5) Pack into an xarray.DataArray
    IVT_regridded = xr.DataArray(
        data,
        dims=("time", "rlat", "rlon"),
        coords={
            "time": IVT_wrapped.time,
            "rlat": rlat,
            "rlon": rlon,
            "lat":  (("rlat","rlon"), lat_tgt),
            "lon":  (("rlat","rlon"), lon_tgt),
        },
        name=IVT_wrapped.name,
        attrs=IVT_wrapped.attrs
    )

    print("✅ Regridding complete: output dims", IVT_regridded.dims)
    return IVT_regridded
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python Regrid_RotPolar_CORDEX.py <YEAR> <MONTH> <INPUT_DIR> <OUTPUT_DIR> <BASE_DIR>")
        sys.exit(1)

    year       = sys.argv[1]
    month      = sys.argv[2]
    input_dir  = sys.argv[3]
    output_dir = sys.argv[4]
    base_dir   = sys.argv[5]

    input_file  = os.path.join(input_dir,   f"IVT_NORESM_{year}{month}.nc")
    output_file = os.path.join(output_dir,  f"IVT_NORESM_CORDEX_{year}{month}.nc")

    print(f"📂 Loading IVT input: {input_file}")
    IVT_org     = xr.open_dataset(input_file)
    IVT         = IVT_org.IVT

    # Activate wrapping
    IVT_wrapped    = wrap_lon(IVT)
    # Activate regridding
    IVT_regridded = regrid_to_ArcticCORDEX(IVT_wrapped, base_dir)

    print(f"💾 Saving regridded IVT to: {output_file}")
    IVT_regridded.to_netcdf(output_file)

    print("✅ Done.")
