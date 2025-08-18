#!/usr/bin/env python
# coding: utf-8
# NorESM

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

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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
    """
    Load and concatenate a variable from a list of NetCDF files.
    Handles aliases like QV -> q, removes duplicates, and prints time range.
    """
    import numpy as np

    VAR_ALIASES = {
        "QV": ["QV", "q"],
        "U": ["U", "u"],
        "V": ["V", "v"]
    }

    possible_names = VAR_ALIASES.get(var_name, [var_name])
    datasets = []

    for f in files:
        try:
            ds = xr.open_dataset(f, engine="h5netcdf", chunks={})
            found = False
            for name in possible_names:
                if name in ds:
                    data = ds[name]
                    if "time" in data.dims:
                        datasets.append(data)
                        found = True
                    else:
                        print(f"⚠️ Variable {name} in {f} has no time dimension.")
                    break
            if not found:
                print(f"⚠️ Variable {var_name} not found in {f}.")
        except Exception as e:
            print(f"❌ Could not load {var_name} from {f}: {e}")

    if datasets:
        combined = xr.concat(datasets, dim="time").sortby("time")

        # 🧹 Remove duplicate timestamps
        _, unique_idx = np.unique(combined["time"], return_index=True)
        combined = combined.isel(time=unique_idx)

        # ✅ Diagnostic info
        start_time = str(combined.time.values[0])
        end_time = str(combined.time.values[-1])
        print(f"📆 {var_name} loaded from {len(files)} files.")
        print(f"📅 Time range: {start_time} → {end_time}")
        print(f"🕒 Total timesteps: {combined.sizes['time']}")
        return combined
    else:
        print(f"⚠️ No valid datasets found for variable '{var_name}'.")
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
        p = np.asarray(p).ravel()
        x = np.asarray(x).ravel()
    
        # Ensure p is ascending for np.interp
        if p[0] > p[-1]:
            p = p[::-1]
            x = x[::-1]
    
        return np.interp(new_p, p, x)

    interpolated = xr.apply_ufunc(
        interp_1d,
        p_full,
        data,
        input_core_dims=[["level"], ["level"]],
        output_core_dims=[["plev"]],
        exclude_dims=set(("level",)),  # tells xarray level disappears
        dask="parallelized",
        vectorize=True,
        dask_gufunc_kwargs={"output_sizes": {"plev": len(target_pressures)}},
        output_dtypes=[data.dtype],
        kwargs={"new_p": target_pressures},
    )

    interpolated = interpolated.assign_coords(plev=target_pressures)
    print(f"Interpolated shape: {interpolated.shape}")
    print(f"Interpolated dims: {interpolated.dims}")

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
            #print(f"Renamed: {old_path} → {new_path}")
    print("renaming for .ncz to .nc completed")

    yyyymm = f"{year}{month:02d}"  # e.g. "198503"
    nc_files = [
        os.path.join(output_dir, f)
        for f in sorted(os.listdir(output_dir))
        if f.startswith("caf") and f.endswith(".nc") and yyyymm in f
    ]
    
    # Load variables QV, U, V from the extracted files
    #print(f"Found {len(nc_files)} files in {archive_path}")
    q_month = load_variable_from_files("QV", nc_files)
    u_month = load_variable_from_files("U", nc_files)
    v_month = load_variable_from_files("V", nc_files)
    ps_month = load_variable_from_files("PS", nc_files)
    
    q_month = q_month.load()
    u_month = u_month.load()
    v_month = v_month.load() 
    ps_month = ps_month.load()
    
    # Subset data for Arctic region (latitude >= LAT_MIN)
    q_month = q_month.where(q_month.lat >= LAT_MIN, drop=True)
    u_month = u_month.where(u_month.lat >= LAT_MIN, drop=True)
    v_month = v_month.where(v_month.lat >= LAT_MIN, drop=True)
    ps_month = ps_month.where(ps_month.lat >= LAT_MIN, drop=True)
    
    print(f"Original variable shape: {v_month.shape}")
    print(f"Original dims: {v_month.dims}")

    # Load hybrid coefficients and surface pressure from one of the files
    sample_file = nc_files[0]
    ds = xr.open_dataset(sample_file, engine="h5netcdf")
    akm = ds["akm"]
    bkm = ds["bkm"]
    
    # Expand hybrid coeffs over full dimensions
    akm_4d = akm.expand_dims({
        "time": ps_month.time,
        "lat":  ps_month.lat,
        "lon":  ps_month.lon
    }).transpose("time", "level", "lat", "lon")
    
    bkm_4d = bkm.expand_dims({
        "time": ps_month.time,
        "lat":  ps_month.lat,
        "lon":  ps_month.lon
    }).transpose("time", "level", "lat", "lon")
    
    # Expand surface pressure to full 4D shape
    ps_4d = ps_month.expand_dims({"level": akm.level}, axis=1).transpose(
        "time", "level", "lat", "lon"
    )
    
    # Compute full pressure field
    p_full = akm_4d + bkm_4d * ps_4d
    
    print("p_full shape:", p_full.shape)
    print("Sample p_full:", p_full.isel(time=0, lat=0, lon=0).values)
    print("p_full min:", p_full.min().compute().item(), "max:", p_full.max().compute().item())


    # Dictionary of variables to process and save
    vars_dict = {'q': q_month, 'u': u_month, 'v': v_month}
    os.makedirs(output_dir, exist_ok=True)

    # Interpolate each variable to standard pressure levels and save to NetCDF
    for var_name, data_var in vars_dict.items():
        data_on_plev = align_and_interpolate(data_var, p_full, target_pressures)

        if not isinstance(data_on_plev.time.values[0], np.datetime64):
            time_np = data_on_plev.indexes["time"].to_datetimeindex()
            data_on_plev = data_on_plev.assign_coords(time=time_np)
        
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
        # reorderd to convention
        ds_out = ds_out.transpose('time', 'plev', 'lat', 'lon')
      
        output_filename = f"{var_name}_{year}{month:02}_Arctic_noresm_9plev.nc"
        output_path = os.path.join(output_dir, output_filename)
        ds_out.to_netcdf(output_path)
        print(f"✓ Saved {var_name} to {output_path}")

    # Close Dask client and cluster
    print("\n✅ Closing Dask client.")
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()


