import numpy as np
import datetime
import os
import xarray as xr
import glob

# Path to your NetCDF files (adjust the path and file pattern)
file_list = glob.glob('./output/ENS-*/fcds_lrfc.nc')
# Open multiple files at once
ds = xr.open_mfdataset(file_list, combine='nested', concat_dim='ens').transpose('ocean_time','ens','channel','lat','lon') # 'lon','lat','channel','ens', 'ocean_time'
n_ens,n_channel,n_time,n_lat,n_lon = [ds.sizes[d] for d in ['ens', 'channel','ocean_time','lat','lon']]
ds["ens"]=(['ens'],np.arange(0,n_ens,1))
ds['ens'].attrs['description'] = "Ensemble members"
del ds['pred'].encoding['coordinates']
ds['pred'].attrs['coordinates'] = "ocean_time, ens, channel, lat, lon"
# get initialization date for file name
fc_dates = ds.variables['ocean_time'][:]
init_timestr = np.datetime_as_string(fc_dates[0],unit='D')

# Define chunking and encoding
chunking = {"ens": 1, "ocean_time": 1, "lat": n_lat, "lon": n_lon, "channel": n_channel}
encoding = {
            "pred": {
                    "dtype": "int16",
                    "_FillValue": -32767,
                    "add_offset":0.,
                    "scale_factor":0.0004,
                    "zlib": True,
                    "complevel": 2,
					"shuffle": True,
                    "chunksizes": tuple(chunking[dim] for dim in ds.dims)
                    },
            "lat": {"dtype": "float32","_FillValue": 1e20},
            "lon": {"dtype": "float32","_FillValue": 1e20},
            "channel": {"dtype": "int32","_FillValue": -32767},
            "ens": {"dtype": "int32","_FillValue": -32767},
            "ocean_time": {"dtype": "float32","_FillValue": 1e20},
            }
# Write to NetCDF
ds.chunk(chunking).to_netcdf(f'./output/fcds_lrfc_ens.nc',format="NETCDF4",encoding=encoding)
ds.close()

os.system(f'cp ./output/fcds_lrfc_ens.nc ./archives/netcdf/fcds_lrfc_ens_{init_timestr}.nc')