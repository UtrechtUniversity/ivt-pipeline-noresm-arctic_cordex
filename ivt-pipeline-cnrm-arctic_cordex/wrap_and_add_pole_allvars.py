import sys
import os
import xarray as xr
import numpy as np

LON_MIN = -180
LON_MAX = 179.999

def select_lon_wrap(ds, lon_min_user, lon_max_user):
    if ds.lon.min() < 0:
        ds = ds.assign_coords(lon=((ds.lon + 360) % 360)).sortby('lon')
    lon_min_0_360 = (lon_min_user + 360) % 360
    lon_max_0_360 = (lon_max_user + 360) % 360

    if lon_min_0_360 <= lon_max_0_360:
        selected_ds = ds.sel(lon=slice(lon_min_0_360, lon_max_0_360))
    else:
        ds1 = ds.sel(lon=slice(lon_min_0_360, 359.99999))
        ds2 = ds.sel(lon=slice(0, lon_max_0_360))
        selected_ds = xr.concat([ds1, ds2], dim='lon').sortby('lon')

    if selected_ds.lon.max() > 180:
        selected_ds['lon'] = xr.where(selected_ds.lon > 180, selected_ds.lon - 360, selected_ds.lon)
        selected_ds = selected_ds.sortby('lon')

    return selected_ds

def add_north_pole_to_dataset(ds, var_name='IVT'):
    lat_dim = 'lat'
    lon_dim = 'lon'
    new_lat = 90.0
    lat_extended = np.append(ds[lat_dim].values, new_lat)
    new_vars = {}

    for name, da in ds.data_vars.items():
        if lat_dim not in da.dims:
            new_vars[name] = da
            continue

        if name == var_name:
            last_lat_slice = da.isel({lat_dim: -1})
            pole_mean = last_lat_slice.mean(dim=lon_dim, keep_attrs=True)
            pole_expanded = pole_mean.expand_dims({lat_dim: [new_lat], lon_dim: ds[lon_dim]})
            pole_expanded = pole_expanded.transpose(*da.dims)
        else:
            shape = list(da.shape)
            lat_index = da.dims.index(lat_dim)
            shape[lat_index] = 1
            fill_value = np.full(shape, np.nan, dtype=da.dtype)
            pole_expanded = xr.DataArray(fill_value, dims=da.dims, coords={dim: da.coords[dim] for dim in da.dims})
            coords = pole_expanded.coords.to_dataset()
            coords[lat_dim] = [new_lat]
            pole_expanded = pole_expanded.assign_coords(coords)

        new_da = xr.concat([da, pole_expanded], dim=lat_dim).sortby(lat_dim)
        new_vars[name] = new_da

    new_ds = xr.Dataset(data_vars=new_vars, coords={**ds.coords, lat_dim: lat_extended})
    new_ds.attrs = ds.attrs
    return new_ds

def process_ivt(year, month, model_input):
    yyyymm = f"{year}{month}"
    input_file = os.path.join(model_input, f"IVT_CNRM_{yyyymm}.nc")
    output_file = os.path.join(model_input, f"IVT_CNRM_{yyyymm}_wrapped_pole.nc")

    print(f"🔍 Loading: {input_file}")
    ds = xr.open_dataset(input_file)
    print(f"📏 Original shape: {ds.lat.shape}, {ds.lon.shape}")

    ds_wrapped = select_lon_wrap(ds, LON_MIN, LON_MAX)
    ds_with_pole = add_north_pole_to_dataset(ds_wrapped, var_name='IVT')

    print(f"✅ New shape: {ds_with_pole.lat.shape}, max lat: {ds_with_pole.lat.values[-1]}")
    ds_with_pole.to_netcdf(output_file)
    print(f"💾 Saved to: {output_file}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python wrap_and_add_pole_allvars.py YEAR MONTH IVT_OUTPUT TRACKABLE_FILES")
        sys.exit(1)

    year = sys.argv[1]
    month = sys.argv[2]
    ivt_output = sys.argv[3]
    trackable_files = sys.argv[4]

    # Then you can pass these variables to your processing function(s) as needed
    # For example:
    process_ivt(year, month, ivt_output)  # or another function that uses trackable_files as well


if __name__ == "__main__":
    main()
