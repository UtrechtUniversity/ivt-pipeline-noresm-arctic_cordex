import numpy as np
import xarray as xr

# Midpoints
plev = np.array([30000, 40000, 50000, 60000, 70000, 75000, 85000, 92500, 100000])

# Infer bounds assuming standard layering (linear in pressure)
plev_bounds = []
for i in range(len(plev)):
    if i == 0:
        lower = plev[i]
        upper = (plev[i] + plev[i+1]) / 2
    elif i == len(plev) - 1:
        lower = (plev[i-1] + plev[i]) / 2
        upper = plev[i]
    else:
        lower = (plev[i-1] + plev[i]) / 2
        upper = (plev[i] + plev[i+1]) / 2
    plev_bounds.append([lower, upper])

plev_bounds = np.array(plev_bounds)  # shape (9, 2)

ds = xr.Dataset(
    {
        "plev_bounds": (("plev", "bounds"), plev_bounds)
    },
    coords={
        "plev": plev,
    },
)

ds["plev_bounds"].attrs["units"] = "Pa"

ds.to_netcdf("level_bounds.nc")
print("✅ level_bounds.nc written")
