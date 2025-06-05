import numpy as np
import datetime
import os
import xarray as xr
import glob
from multiprocessing import Pool, cpu_count

def save_nc(args):
    iens,init_timestr = args
    # Path to your NetCDF files
    file_name = f'./output/ENS-{iens}/fcds_lrfc.nc'
    # Open file
    ds0 = xr.open_dataset(file_name).transpose('ocean_time','channel','lat','lon') # 'lon','lat','channel','ens', 'ocean_time'
    n_channel,n_time,n_lat,n_lon = [ds0.sizes[d] for d in ['channel','ocean_time','lat','lon']]

    ds = xr.Dataset(
        data_vars=dict(
            SSH=(['ocean_time','lat','lon'], ds0['pred'][:,2,:,:].values,{'units':'m','discription':'sea surface height'}),
            SSU=(['ocean_time','lat','lon'], ds0['pred'][:,0,:,:].values,{'units':'m/s','discription':'sea surface velocity, u component'}),
            SSV=(['ocean_time','lat','lon'], ds0['pred'][:,1,:,:].values,{'units':'m/s','discription':'sea surface velocity, v component'}),
            SSKE=(['ocean_time','lat','lon'], ds0['pred'][:,3,:,:].values,{'units':'m^2/s^2','discription':'sea surface kinetic energy (u^2+v^2)'}),
            ),
        coords=dict(
            lon=("lon", ds0['lon'].values,{'units':'degree_east'}),
            lat=("lat", ds0['lat'].values,{'units':'degree_north'}),
            ocean_time=("ocean_time", ds0['ocean_time'].values),
            ),
        attrs=dict(description="OceanNet 2.0 ensemble forecast"),
        )

    # Define chunking and encoding
    chunking = {"ocean_time": n_time, "lat": n_lat, "lon": n_lon}
    encoding = {
                "SSH": {
                        "dtype": "int16",
                        "_FillValue": -32767,
                        "add_offset":0.,
                        "scale_factor":0.0004,
                        "zlib": True,
                        "complevel": 2,
                        "shuffle": True,
                        "chunksizes": tuple(chunking[dim] for dim in ds['SSH'].dims)
                        },
                "SSU": {
                        "dtype": "int16",
                        "_FillValue": -32767,
                        "add_offset":0.,
                        "scale_factor":0.0004,
                        "zlib": True,
                        "complevel": 2,
                        "shuffle": True,
                        "chunksizes": tuple(chunking[dim] for dim in ds['SSU'].dims)
                        },
                "SSV": {
                        "dtype": "int16",
                        "_FillValue": -32767,
                        "add_offset":0.,
                        "scale_factor":0.0004,
                        "zlib": True,
                        "complevel": 2,
                        "shuffle": True,
                        "chunksizes": tuple(chunking[dim] for dim in ds['SSV'].dims)
                        },
                "SSKE": {
                        "dtype": "int16",
                        "_FillValue": -32767,
                        "add_offset":0.,
                        "scale_factor":0.0004,
                        "zlib": True,
                        "complevel": 2,
                        "shuffle": True,
                        "chunksizes": tuple(chunking[dim] for dim in ds['SSKE'].dims)
                        },
                "lat": {"dtype": "float32","_FillValue": 1e20},
                "lon": {"dtype": "float32","_FillValue": 1e20},
                "ocean_time": {"dtype": "float32","_FillValue": 1e20,'units':'days since 1970-01-01 00:00:00'},
                }
    # Write to NetCDF
    ds.chunk(chunking).to_netcdf(f'./nc4vis/{init_timestr}/oceannet_ens_{iens}.nc',format="NETCDF4",encoding=encoding)

if __name__ == '__main__':
    # Path to your NetCDF files
    file_name = f'./output/ENS-0/fcds_lrfc.nc'
    # Open file
    ds0 = xr.open_dataset(file_name)
    # get initialization date for file name
    init_timestr = np.datetime_as_string(ds0.variables['ocean_time'][0],unit='D').replace('-','')
    os.makedirs(f'./nc4vis/{init_timestr}/', exist_ok=True)

    log_parallel=1 # logic swith to turn on/off parallel plot (0=sequential, 1=parallel)
    # Prepare arguments for parallel processing
    args = [(iens, init_timestr) for iens in range(40)]
    if (log_parallel == 0):
        for iarg in args:
            save_nc(iarg)
    elif (log_parallel == 1):
        num_processes = min(cpu_count(), len(args))  # Use number of available cores or number of frames, whichever is smaller
        # Create a pool of workers and process frames in parallel
        with Pool(processes=num_processes) as ipool:
            ipool.map(save_nc, args)