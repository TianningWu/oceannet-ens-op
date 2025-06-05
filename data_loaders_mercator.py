import numpy as np
import netCDF4 as nc
import torch
from torch.nn.functional import interpolate
import gc
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
import datetime

data_dir = "/ourdisk/hpc/ai2es/twu27/srcdata/"

torch.manual_seed(0)
np.random.seed(0)

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache

def numpy_to_cuda(arr):
    return torch.from_numpy(arr).float().cuda()

def cuda_to_numpy(arr):
    return arr.cpu().detach().numpy()

def zero_mask(arrs):
    ## Givin list of arrays with nan values representing masked regions
    ##  finds the overlapping masks, and returns masked numpy arrays with the same dimension and masks
    
    if type(arrs) is not type([]):
        arrs = [arrs]

    shared_mask = np.sum([np.isnan(arr) for arr in arrs], axis = 0)
    arrs_ma = [np.ma.masked_array(np.where(shared_mask, 0, arr), shared_mask) for arr in arrs]
    
    return arrs_ma

## time dict, to arange the times in a straightforward way to pull from the correct dataset without having to save
EARTHR = 6378.137

CONVKERNEL1 = np.array([[-1.000,-1.414,-1.000],
                        [-1.414,+9.657,-1.414],
                        [-1.000,-1.414,-1.000]])/(9.657) # image sharpening kernel -> sharpens contrasting points, boldening high peaking regions

# CONVKERNEL1_torch = numpy_to_cuda(np.tile(CONVKERNEL1,(1,1,1)))

def area_element_angles(dlat, dlon, lat, lon, r = EARTHR, degrees = True):
    if degrees:
        return r**2 * np.sin((90-lat) * np.pi/180) * dlat * np.pi/180 * dlon* np.pi/180
    else:
        return r**2 * np.sin(lat) * dlat * dlon

area_element_angles_v = np.vectorize(area_element_angles)

def area_elements_lats_lons(lats, lons, r = EARTHR, degrees = True):
    ## assumes that lat and lon spacing is small
    # lats = 90 - np.array(lats) # converting to degrees from the vertical
    # lons = np.array(lons)
    X, Y = np.meshgrid(lons, lats)
    # average lon/lat distance between adjacent lon/lat coordinates
    dlons = np.convolve(lons, [1,0,-1], mode = "same")/2
    dlats = np.convolve(lats, [1,0,-1], mode = "same")/2

    # duplicating previous to last dlon/ldat coordinates for boundaries
    dlons[0] = dlons[1]
    dlons[-1] = dlons[-2]
    dlats[0] = dlats[1]
    dlats[-1] = dlats[-2]

    # mesh grids of what dlon/dlat look like at a specific lon/lat coordinate
    dX, dY = np.abs(np.meshgrid(dlons, dlats))

    return area_element_angles_v(dY, dX, Y, X, r = r, degrees = True)

def mercator_daily_data2():
    num_days_mercator = 0
    time_dict_mercator = {}
    t = 0

    ds = nc.Dataset(f"{data_dir}/Mercator_hourly/Mercator_IC.nc")
    times1 = ds["time"][:].data.tolist()
    date0 = datetime.datetime.strptime("1950/1/1", "%Y/%m/%d")
    times1_datetime = [datetime.timedelta(hours=i) + date0 for i in times1]
	
    for iday in times1_datetime:
        time_dict_mercator[t] = (iday.year,iday.timetuple().tm_yday) 
        t+=1
    num_days_mercator+=len(times1)
    
    rev_time_dict_mercator = {v:k for [k,v] in time_dict_mercator.items()} 
    
    return time_dict_mercator, rev_time_dict_mercator, num_days_mercator

time_dict_mercator, rev_time_dict_mercator, num_days_mercator = mercator_daily_data2()

def load_mercator_whole2(steps, 
                        time_dict = time_dict_mercator, 
                        channels = ["SSU","SSV","SSH", "SSKE"],
                        ):
    
    ## since all the initial values must be added first 
    numchannels = len(channels)
    
    gloryshr = nc.Dataset(f"{data_dir}/Mercator_hourly/Mercator_IC.nc")
    
    lrcoordmeshX, lrcoordmeshY = np.meshgrid(gloryshr["longitude"][:], gloryshr["latitude"][:])
    lrcoordmeshX, lrcoordmeshY = lrcoordmeshX.data, lrcoordmeshY.data
    lrloncoords = lrcoordmeshX[0,:]
    lrlatcoords = lrcoordmeshY[:,0]
    
    g1 = gloryshr["zos"][0,0,:,:]
    lr = np.empty((len(steps),g1.shape[0],g1.shape[1],numchannels))
    times=[]
    for it, t in enumerate(steps,0):
        times.append(time_dict[t])
        for ich, ch in enumerate(channels,0):
            if ch == "SSU":
                lr[it,:,:,ich] = gloryshr["uo"][t,0,:,:].filled(fill_value=np.nan)
            if ch == "SSV":
                lr[it,:,:,ich] = gloryshr["vo"][t,0,:,:].filled(fill_value=np.nan)
            if ch == "SSH":
                lr[it,:,:,ich] = gloryshr["zos"][t,0,:,:].filled(fill_value=np.nan)
            ## add_channels
            if ch == "SSKE":
                uind = channels.index("SSU")
                vind = channels.index("SSV")
                lr[it,:,:,ich] = lr[it,:,:,uind]**2+lr[it,:,:,vind]**2
    
    gloryshr.close()
    
    return lr, lrloncoords, lrlatcoords, times

def load_predict_grid_mask(lat_lon_keep = (17.1, 30.9, -98.0, -74.1)):

    romshr_grid = nc.Dataset(f"{data_dir}/CNAPS2/cnaps2_wholegom_grid.nc")
    hrcoordmeshX, hrcoordmeshY = romshr_grid["lon_rho"][:].data, romshr_grid["lat_rho"][:].data
    hrmask = romshr_grid["mask_rho"][:]
    hrh = romshr_grid["h"][:]
    romshr_grid.close()

    # for comparing same regions between lr and hr
    (latmin, latmax, lonmin, lonmax) = lat_lon_keep
    hrdomain = ((latmin < hrcoordmeshY) * (hrcoordmeshY < latmax) * (lonmin < hrcoordmeshX) * (hrcoordmeshX < lonmax))

    hrdomain_lonKeep = hrdomain.any(axis = 0)
    hrdomain_latKeep = hrdomain.any(axis = 1)

    hrmask_crop = hrmask[hrdomain_latKeep, :]
    hrmask_crop = hrmask_crop[:, hrdomain_lonKeep]
    hrh_crop = hrh[hrdomain_latKeep, :]
    hrh_crop = hrh_crop[:, hrdomain_lonKeep]
    
    loncoords = hrcoordmeshX[0,hrdomain_lonKeep]
    latcoords = hrcoordmeshY[hrdomain_latKeep,0]
    
    return hrmask_crop, loncoords, latcoords, hrh_crop


def add_land_sea_mask(data, mask):
    return torch.concat([data, mask], axis = -1)

def lat_lon_interpolate(data, latsog, lonsog, latsnew, lonsnew, interpolator_use = "scipy"):
    ## bicubic interpolation, torch
    if interpolator_use == "torch":
        data_interp = interpolate(torch.from_numpy(data).permute(0,3,1,2), data.shape[1:3], mode = "bicubic").permute(0,2,3,1).numpy()
    
    ## step x lat x lon x channel
    elif interpolator_use == "scipy":
        X, Y = np.meshgrid(lonsnew, latsnew)
        # print(lr_crop.shape)
        # print(Y.shape, X.shape)
        data_interp = np.empty((data.shape[0], len(latsnew), len(lonsnew), data.shape[3]))
        ## each time step and channel needs to be separated, so cant do each interpolation at once.
        for istep in range(data_interp.shape[0]):
            for ivar in range(data_interp.shape[3]):                     
                [data_step_var] = zero_mask([data[istep,:,:,ivar]])                 
                interpolator = RegularGridInterpolator((latsog, lonsog), data_step_var.data, method = "linear")

                data_interp[istep,:,:,ivar] = interpolator((Y,X))

    return data_interp