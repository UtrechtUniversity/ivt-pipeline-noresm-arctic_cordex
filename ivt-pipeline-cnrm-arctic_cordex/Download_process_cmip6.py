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


# Chunking functions
def generate_chunks_hus(year):
    return [
        (f"{year}01010600", f"{year}07010000"),
        (f"{year}07010600", f"{year+1}01010000"),
    ]

def generate_chunks_ua_va(year):
    return [
        (f"{year}01010600", f"{year}05010000"),
        (f"{year}05010600", f"{year}09010000"),
        (f"{year}09010600", f"{year+1}01010000"),
    ]

chunk_generators = {
    'hus': generate_chunks_hus,
    'ua': generate_chunks_ua_va,
    'va': generate_chunks_ua_va,
        }

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
    data = data.chunk({'lev': -1})
    p_full = p_full.chunk({'lev': -1})

    def interp_1d(p, x, new_p):
        return np.interp(new_p, p[::-1], x[::-1])  # reverse for ascending order

    interpolated = xr.apply_ufunc(
        interp_1d,
        p_full,
        data,
        input_core_dims=[["lev"], ["lev"]],
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
    common_times = np.intersect1d(data_var.time.values, p_full.time.values)
    data_sel = data_var.sel(time=common_times)
    p_sel = p_full.sel(time=common_times)
    return interpolate_to_pressure_levels(data_sel, p_sel, target_pressures)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Process CMIP6 data")
    parser.add_argument("--gcm", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Fetch resources from SLURM environment variables
    n_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', 8))  # fallback 8
    mem_bytes = int(os.environ.get('SLURM_MEM_PER_NODE', 128 * 1024 ** 3))  # fallback 128GB in bytes
    mem_gb = mem_bytes // (1024 ** 3)
    
    # Decide number of workers and threads per worker (adjust as needed)
    n_workers = 4
    threads_per_worker = max(1, n_cpus // n_workers)
    mem_per_worker = int(mem_gb * 0.85 // n_workers)  # 85% of mem divided by workers
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=f'{mem_per_worker}GB'
    )
    client = Client(cluster)
    print(f"Dask client dashboard link: {client.dashboard_link}")
    # You should see information about your workers when you run client
    print(client)
    
    # Optional: Set Dask network timeout (can help with remote data stability)
    dask.config.set({'distributed.comm.tcp.timeout.connect': '60s'})
    dask.config.set({'distributed.comm.tcp.timeout.read': '300s'})
    
    
    # === USER CONFIGURATION ===
    year = args.year
    month = args.month
    
    # Scenario switching configuration
    scenario_start_year = 2015
    
    # Target pressure levels in Pa (convert from hPa)
    target_pressures = np.array([300, 400, 500, 600, 700, 750, 850, 925, 1000], dtype=float) * 100

    # Latitude threshold for Arctic region subset
    LAT_MIN=42

    # ESGF URL templates for each experiment
    experiment_config = {
    "historical": {
        "base_url_template": "https://esgf.ceda.ac.uk/thredds/dodsC/esg_cmip6/CMIP6/CMIP/CNRM-CERFACS/CNRM-ESM2-1/{experiment}/r1i1p1f2/6hrLev/",
        "end_url": "/gr/v20181206/"
    },
    "ssp370": {
        "base_url_template": "http://esg1.umr-cnrm.fr/thredds/dodsC/CMIP6_CNRM/ScenarioMIP/CNRM-CERFACS/CNRM-ESM2-1/{experiment}/r1i1p1f2/6hrLev/",
        "end_url": "/gr/v20191021/"
    }
    }
    
    file_template = "{var}_6hrLev_CNRM-ESM2-1_{experiment}_r1i1p1f2_gr_{start}-{end}.nc"


    ######################## Main processing loop ################################

    print(f"\n=== Processing {year}-{month:02} ===")

    current_month_start_str = f"{year}-{month:02}-01"
    if month == 12:
            next_month_start_str = f"{year+1}-01-01"
    else:
            next_month_start_str = f"{year}-{month+1:02}-01"

    experiment = "historical" if year < scenario_start_year else "ssp370"
    config = experiment_config[experiment]

    q_month, u_month, v_month = None, None, None
    last_va_url_for_pressure_vars = None # To store a valid URL for fetching ap, b

    for var in ['hus', 'ua', 'va']:
            chunks = chunk_generators[var](year)
            var_data_for_month_list = []

            for chunk_start, chunk_end in chunks:
                    try:
                            chunk_start_dt = pd.to_datetime(chunk_start, format="%Y%m%d%H%M")
                            chunk_end_dt = pd.to_datetime(chunk_end, format="%Y%m%d%H%M")
                            target_period_start_dt = pd.to_datetime(current_month_start_str)
                            target_period_end_dt = pd.to_datetime(next_month_start_str)

                            if not (chunk_end_dt > target_period_start_dt and chunk_start_dt < target_period_end_dt):
                                    continue

                            base_url = config["base_url_template"].format(experiment=experiment)
                            file_name = file_template.format(var=var, experiment=experiment, start=chunk_start, end=chunk_end)
                            url = base_url + var + config["end_url"] + file_name

                            if var == 'va': # Store the last va url, assuming ap/b are in va files
                                    last_va_url_for_pressure_vars = url

                            print(f"→ Opening {var} from {url} for {year}-{month:02}")

                            ds_chunk = None 
                            try:
                                    #ds_chunk = xr.open_dataset(url, chunks={'time': 'auto'})
                                    # Ensure chunks are applied across all relevant dimensions for parallel processing
                                    ds_chunk = xr.open_dataset(url, chunks={'time': 'auto', 'lat': 'auto', 'lon': 'auto', 'lev': 'auto'}) # ADDED/ALTERED: 'lat', 'lon', 'lev' chunks
                                    effective_slice_start = max(chunk_start_dt, target_period_start_dt)
                                    effective_slice_end = min(chunk_end_dt, target_period_end_dt)

                                    # Create string representations for slicing
                                    slice_start_str = effective_slice_start.strftime('%Y-%m-%d %H:%M:%S')
                                    # For end slice, if it's the start of the next month, make it exclusive
                                    if effective_slice_end == target_period_end_dt:
                                             slice_end_str = (effective_slice_end - pd.Timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
                                    else:
                                             slice_end_str = effective_slice_end.strftime('%Y-%m-%d %H:%M:%S')

                                    # select time slice (month of interest in year of interest)
                                    ds_selected_month_in_chunk = ds_chunk.sel(time=slice(slice_start_str, slice_end_str))
                                    # select arctic region (>45deg North)
                                    ds_selected_month_in_chunk = ds_selected_month_in_chunk.where(ds_selected_month_in_chunk.lat >= LAT_MIN, drop=True)


                                    if not ds_selected_month_in_chunk.time.size:
                                            print(f"  No data for {year}-{month:02} in this chunk of {var}.")
                                            continue 

                                    ds_selected_month_in_chunk = ds_selected_month_in_chunk.sortby(['time', 'lat', 'lon'])
                                    var_data_for_month_list.append(ds_selected_month_in_chunk[var])
                            except Exception as e:
                                    print(f"❌ Failed loading or processing {var} from {url}: {e}")
                            finally:
                                    if ds_chunk is not None: ds_chunk.close()
                    except Exception as e:
                            print(f"❌ Error parsing chunk dates or general chunk processing for {var}: {e}")

            if var_data_for_month_list:
                    concatenated_var_data = xr.concat(var_data_for_month_list, dim='time').sortby('time')
                    _, index = np.unique(concatenated_var_data['time'], return_index=True)
                    concatenated_var_data = concatenated_var_data.isel(time=index)
                    if var == 'hus': q_month = concatenated_var_data
                    elif var == 'ua': u_month = concatenated_var_data
                    elif var == 'va': v_month = concatenated_var_data
            else:
                    print(f"⚠️ No data found for {var} for {year}-{month:02} after processing all chunks.")

    with xr.open_dataset(last_va_url_for_pressure_vars, chunks={'time':'auto','lat':'auto','lon':'auto'}) as ds_p:
            ps = ds_p['ps'].sel(time=slice(current_month_start_str, next_month_start_str))
            ps = ps.where(ps.lat >= LAT_MIN, drop=True)
            ap = ds_p['ap']
            b = ds_p['b']

            # Expand dims for broadcasting
            ap_4d = ap.expand_dims({'time': ps.time, 'lat': ps.lat, 'lon': ps.lon}).transpose('time', 'lev', 'lat', 'lon')
            b_4d = b.expand_dims({'time': ps.time, 'lat': ps.lat, 'lon': ps.lon}).transpose('time', 'lev', 'lat', 'lon')
            ps_4d = ps.expand_dims({'lev': ap_4d.lev}, axis=1).transpose('time', 'lev', 'lat', 'lon')

            p_full = ap_4d + b_4d * ps_4d

            vars_dict = {'q': q_month, 'u': u_month, 'v': v_month}
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
                    output_dir = args.output_dir  # use the directory passed as argument
                    os.makedirs(output_dir, exist_ok=True)  # create it if it doesn't exist
                    output_filename = f"{var_name}_{year}{month:02}_Arctic_cnrm_9plev.nc"
                    output_path = os.path.join(output_dir, output_filename)
                    ds_out.to_netcdf(output_path)
                    print(f"Saved {var_name} data to {output_path}")

    # Close the Dask client when you're done with all processing
    print("\nClosing Dask client.")
    client.close()
    cluster.close()

if __name__ == "__main__":
        main()

