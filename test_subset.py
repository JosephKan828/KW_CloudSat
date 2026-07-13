from pathlib import Path
import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np

def _get_subset(fname: Path) -> tuple[dict, np.ndarray]:
    with xr.open_dataset(fname) as ds_xr:
        time_vals = pd.to_datetime(ds_xr['time'].values)
        lat_vals = ds_xr['lat'].values
        lon_vals = ds_xr['lon'].values
        lev_name = 'lev' if 'lev' in ds_xr.coords else 'level'
        lev_vals = ds_xr[lev_name].values
        
    t_mask = (time_vals >= pd.Timestamp("2006-01-01")) & (time_vals <= pd.Timestamp("2017-12-31"))
    t_indices = np.where(t_mask)[0]
    
    lat_mask = (lat_vals >= -5) & (lat_vals <= 5)
    lat_indices = np.where(lat_mask)[0]
    
    t_start, t_end = t_indices[0], t_indices[-1] + 1
    lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1

    with nc.Dataset(fname, "r") as ds:
        var_keys = [
                key for key in ds.variables.keys()
                if key not in ds.dimensions.keys()
                ]
        var = ds.variables[var_keys[0]]

        slices = []
        for dim in var.dimensions:
            if dim == 'time':
                slices.append(slice(t_start, t_end))
            elif dim == 'lat':
                slices.append(slice(lat_start, lat_end))
            else:
                slices.append(slice(None))
                
        subset = var[tuple(slices)]
 
        lat_axis = var.dimensions.index('lat')
        var_global = np.nanmean(subset, axis=lat_axis)
        
        time_axis = var.dimensions.index('time')
        if time_axis > lat_axis:
            time_axis -= 1

    coords_global = {
        "lon": lon_vals,
        "lev": lev_vals,
        "time": time_vals[t_start:t_end]
    }

    anom_global = (var_global - np.nanmean(var_global, axis=time_axis, keepdims=True))

    return coords_global, anom_global

if __name__ == "__main__":
    coords, anom = _get_subset(Path('/data92/b11209013/ERA5/w/w_Itp_sub.nc'))
    print("Coords:")
    for k, v in coords.items(): print(f"  {k}: shape {v.shape}")
    print("Anom shape:", anom.shape)

