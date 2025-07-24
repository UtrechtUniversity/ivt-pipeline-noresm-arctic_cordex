#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import os
from dask.distributed import Client, LocalCluster
import dask 
import argparse
import tarfile
import glob
import shutil
import tempfile
import sys
import subprocess

def copy_from_ecfs(archive_path, local_dir):
    """Copy a .tar file from ECFS to a local directory using `ecp`."""
    basename = os.path.basename(archive_path)
    local_path = os.path.join(local_dir, basename)

    if os.path.exists(local_path):
        print(f"✓ Archive already copied locally: {local_path}")
        return local_path

    print(f"→ Starting ECFS copy...")
    print(f"   Source: {archive_path}")
    print(f"   Destination: {local_path}")

    result = subprocess.run(["ecp", archive_path, local_path])

    if result.returncode != 0:
        raise RuntimeError(f"❌ Failed to copy file from ECFS: {archive_path}")
    
    print(f"✓ ECFS copy complete.")
    return local_path


def extract_tar(archive_path, extract_dir):
    """Extracts a tar file to the given directory and returns list of extracted files."""
    print(f"→ Extracting {archive_path} to {extract_dir}")
    extracted_files = []
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=extract_dir)
        # Collect full paths of extracted regular files
        extracted_files = [os.path.join(extract_dir, member.name) for member in tar.getmembers() if member.isfile()]
    print(f"✓ Extraction complete.")
    return extracted_files

def load_variable_from_files(var_name, files):
    """Load and concatenate a variable from a list of NetCDF files."""
    datasets = []
    for f in files:
        try:
            ds = xr.open_dataset(f, engine="netcdf4",chunks={})#{'time': 'auto'})
            if var_name in ds:
                datasets.append(ds[var_name])
        except Exception as e:
            print(f"⚠️ Could not load {var_name} from {f}: {e}")
    if datasets:
        return xr.concat(datasets, dim='time').sortby('time')
    else:
        return None
        
def interpolate_to_pressure_levels(data, p_full, target_pressures):
    """
    Interpolate data (e.g., q, u, v) from model pressure levels to standard pressure levels.

    Parameters:
    - data: xarray.DataArray with dims (time, lev, lat, lon)
    - p_full: pressure field with same dims (time, lev, lat, lon)
    - target_pressures: 1D numpy array of pressure levels (in Pa) to interpolate to

    Returns:
    - Interpolated data with dims (time, plev, lat, lon)
    """
    # Rechunk lev dimension to a single chunk
    data = data.chunk({'level': -1})
    p_full = p_full.chunk({'level': -1})

    def interp_1d(p, x, new_p):
        return np.interp(new_p, p[::-1], x[::-1])  # reverse for ascending order

    interpolated = xr.apply_ufunc(
        interp_1d,
        p_full,
        data,
        input_core_dims=[["level"], ["level"]],
        output_core_dims=[["plev"]],
        output_sizes={"plev": len(target_pressures)},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
        kwargs={"new_p": target_pressures},
    )

    interpolated = interpolated.assign_coords(plev=target_pressures)
    return interpolated

def align_and_interpolate(data_var, p_full, target_pressures):
    """
    Aligns time coordinates of data and pressure arrays, then interpolates data to target pressures.

    Ensures both inputs share the same time steps by intersecting their time coordinates
    to avoid dimension mismatches during interpolation.
    """
    common_times = np.intersect1d(data_var.time.values, p_full.time.values)
    data_sel = data_var.sel(time=common_times)
    p_sel = p_full.sel(time=common_times)
    return interpolate_to_pressure_levels(data_sel, p_sel, target_pressures)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process NorESM archive data")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--archive_in", type=str, required=True)
    args = parser.parse_args()
    print("Arguments received:", sys.argv)
    
    # Extract variables from args
    year = args.year
    month = args.month
    output_dir = args.output_dir
    archive_in = args.archive_in

    # Setup Dask distribution client for parallel processing
    n_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))
    mem_bytes = int(os.environ.get('SLURM_MEM_PER_NODE', 128 * 1024 ** 3))
    mem_gb = mem_bytes // (1024 ** 3)
    n_workers = 4
    threads_per_worker = max(1, n_cpus // n_workers)
    mem_per_worker = int(mem_gb * 0.85 // n_workers)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=f'{mem_per_worker}GB'
    )
    client = Client(cluster)
    print(f"Dask client: {client.dashboard_link}")

    # Set some Dask communication timeouts for stability
    dask.config.set({'distributed.comm.tcp.timeout.connect': '60s'})
    dask.config.set({'distributed.comm.tcp.timeout.read': '300s'})

    # Determine experiment name and archive tar file path based on year and month
    scenario_start_year = 2015
    experiment = "historical" if year < scenario_start_year else "ssp370"
    tar_filename = f"NorESM2-MM_{experiment}_r1i1p1f1_{year}_{month:02}.tar"
    archive_path = os.path.join(archive_in, tar_filename)
    
    print(f"Processing archive file: {archive_path}")    
    
    # Define standard pressure levels for interpolation and latitude cutoff for Arctic region
    target_pressures = np.array([300, 400, 500, 600, 700, 750, 850, 925, 1000], dtype=float) * 100
    LAT_MIN = 42

    # === Copy the tar archive locally and extract its contents ===
    print(f" Preparing to fetch and extract from archive: {archive_path}")
    local_tar = copy_from_ecfs(archive_path, output_dir)
    print(f" Extracting contents from: {local_tar}")
    extracted_files = extract_tar(local_tar, output_dir)
    print(f"✓ Extraction done.")
    
    # Rename .ncz files to .nc after extraction
    for fname in os.listdir(output_dir):
        if fname.endswith(".ncz"):
            old_path = os.path.join(output_dir, fname)
            new_path = os.path.join(output_dir, fname.replace(".ncz", ".nc"))
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} → {new_path}")

    nc_files = sorted(glob.glob(os.path.join(output_dir, "*.nc")))
    
    # Load variables QV, U, V from the extracted files
    print(f"Found {len(nc_files)} files in {archive_path}")
    q_month = load_variable_from_files("QV", nc_files)
    u_month = load_variable_from_files("U", nc_files)
    v_month = load_variable_from_files("V", nc_files)

    if q_month is None or u_month is None or v_month is None:
        print("❌ Failed to load all required variables (QV, U, V). Exiting.")
        return

    # Subset data for Arctic region (latitude >= LAT_MIN)
    q_month = q_month.where(q_month.lat >= LAT_MIN, drop=True)
    u_month = u_month.where(u_month.lat >= LAT_MIN, drop=True)
    v_month = v_month.where(v_month.lat >= LAT_MIN, drop=True)

    # Load hybrid coefficients and surface pressure from one of the files
#    with xr.open_dataset(nc_files[0], chunks={'time': 'auto'}, engine="netcdf4") as ds:
    with xr.open_dataset(nc_files[0],engine="netcdf4",chunks={}) as ds:
        ps = ds['PS'].where(ds.lat >= LAT_MIN, drop=True)
        akm = ds['akm']  # size 32 (levels)
        bkm = ds['bkm']  # size 32 (levels)

        # Expand dims for broadcasting: time, lat, lon from PS, level from akm/bkm
        akm_4d = akm.expand_dims({'time': ps.time, 'lat': ps.lat, 'lon': ps.lon}).transpose('time', 'level', 'lat', 'lon')
        bkm_4d = bkm.expand_dims({'time': ps.time, 'lat': ps.lat, 'lon': ps.lon}).transpose('time', 'level', 'lat', 'lon')
        ps_4d = ps.expand_dims({'level': akm.level}, axis=1).transpose('time', 'level', 'lat', 'lon')

        # Calculate full pressure at model levels
        p_full = akm_4d + bkm_4d * ps_4d

    # Dictionary of variables to process and save
    vars_dict = {'q': q_month, 'u': u_month, 'v': v_month}
    os.makedirs(output_dir, exist_ok=True)

    # Interpolate each variable to standard pressure levels and save to NetCDF
    for var_name, data_var in vars_dict.items():
        data_on_plev = align_and_interpolate(data_var, p_full, target_pressures)
        ds_out = xr.Dataset({var_name: data_on_plev})
        ds_out = ds_out.assign_coords({
            'plev': target_pressures,
            'time': data_on_plev.time,
            'lat': data_on_plev.lat,
            'lon': data_on_plev.lon,
        })
        ds_out['plev'].attrs.update({
            'units': 'Pa',
            'standard_name': 'air_pressure',
            'axis': 'Z',
            'positive': 'down'
        })
        output_filename = f"{var_name}_{year}{month:02}_Arctic_noresm_9plev.nc"
        output_path = os.path.join(output_dir, output_filename)
        ds_out.to_netcdf(output_path)
        print(f"✓ Saved {var_name} to {output_path}")

    # clean up tar and nc(z) files after processing
    try:
        os.remove(local_tar)
        print(f"✓ Deleted archive file: {local_tar}")
    except Exception as e:
        print(f"⚠️ Could not delete archive file {local_tar}: {e}")
    
    for f in extracted_files:
        try:
            os.remove(f)
            print(f"✓ Deleted extracted file: {f}")
        except Exception as e:
            print(f"⚠️ Could not delete extracted file {f}: {e}")


    # Close Dask client and cluster
    print("\n✅ Closing Dask client.")
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()


