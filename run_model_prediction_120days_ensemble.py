import numpy as np
from netCDF4 import Dataset
import torch
from torch.nn.functional import interpolate
import os
import models
from models import numpy_to_cuda, cuda_to_numpy, STEPS_SCHEMES
import yaml
from data_loaders_mercator import load_mercator_whole2, add_land_sea_mask
import time

torch.manual_seed(0)
np.random.seed(0)

## data loader
mercator_data = load_mercator_whole2

#############################
### LOAD FC MODEL ###
#############################
channels_fc = ["SSU", "SSV", "SSH", "SSKE"]

## loading forecast model
_, lrlons, lrlats, _ = mercator_data([0], channels = channels_fc) # gets appropriate latitude and longitude coordinates for fno2d grid. #lrlons. lrlats are 1d arrays
fc_model_loc = "/ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op/model/model.pt"
fc_model_config_loc = "/ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op/model/config.yml"

with open(fc_model_config_loc, "r") as h:
     config_dict_fc = yaml.load(h, Loader = yaml.Loader)

stepscheme_fc = config_dict_fc["stepscheme"] # time stepping method
# populate nn archtecture into "net_fc"
if config_dict_fc["landseamask"]: # whether include land-sea mask as a channel into the model
    net_fc = getattr(models, config_dict_fc["nn_arch"])(channels = len(channels_fc)+1, channelsout = len(channels_fc), gridx = lrlats, gridy = lrlons, **config_dict_fc["nn_params"]).to("cuda")
    # getattr(models, config_dict_fc["nn_arch"]) is as models.config_dict_fc["nn_arch"]; .to("cuda") move data to GPU
else:
    net_fc = getattr(models, config_dict_fc["nn_arch"])(channels = len(channels_fc), channelsout = len(channels_fc), gridx = lrlats, gridy = lrlons, **config_dict_fc["nn_params"]).to("cuda")
net_fc.load_state_dict(torch.load(fc_model_loc)) # populate weights into "net_fc"
net_fc.eval() # switch to evaluation mode, fix all weight. It is not training

##############################
##### CONDUCT FC ######
##############################

_, lrlons, lrlats, times = mercator_data([28], channels = channels_fc)# loading invariants

steps_autoreg = 120+1 # +1 is for the first frame, which is initial condition

## deal with forecast date and time
date0 = np.datetime64('1858-11-17', 'D')

datetime_fc=np.empty([steps_autoreg,1],dtype='float')
for iday in range(steps_autoreg):
    datetime_fc[iday] = np.datetime64(f'{times[0][0]}-01-01') + np.timedelta64(times[0][1]-1+iday, 'D') - date0 # times[0][1]-1: Jan 1 is the 1st day

for teststart in np.arange(0,40): # forecast initial date
    ## loading data
    lrvtest, _, _, _ = mercator_data([teststart], channels = channels_fc)
    
    maskfc = np.isnan(lrvtest[[0],:,:,:]) #3d mask: shape=[lat, lon, channel]
    lrvmask = np.isnan(lrvtest) #4d mask: shape=[time, lat, lon, channel]
    lrvtest = np.where(np.isnan(lrvtest), 0, lrvtest) # replace all nan with 0
    lrvtest = np.ma.masked_array(lrvtest, lrvmask) # mask lrvtest according to lrvmask (0 & 1). shape=[time, lat, lon, channel] 

    ## generate forecast matrix and provide initial condition
    preds_fc = np.empty([steps_autoreg, lrvtest.shape[1], lrvtest.shape[2], len(channels_fc)]) # create an empty matrix to put forecast data. shape=[fc_timesteps, lrlat, lrlon, fc channel] 
    preds_fc[0,...] = np.nan_to_num(lrvtest[0,...]) # put the first frame of lrvtest into the first frame of preds_fc, as a the initial condition. Also NaN is replaced by zero.

    ## changing expected grid rectilinear lat/lon coordinates
    net_fc.gridx = lrlats
    net_fc.gridy = lrlons

    print("Autoregression, FC...")
    ## autoregressive prediction, with 0 land masking and boolean land/sea channel
    for i in range(1, steps_autoreg): # start from the second frame, because the first frame is initial condition
        inputfc = numpy_to_cuda(preds_fc[[i-1],...]) # input of each time step, i.e. the previous frame of preds_fc, and put data onto GPU

        if config_dict_fc["landseamask"]:
            inputfc = add_land_sea_mask(inputfc, numpy_to_cuda(maskfc[...,[0]].astype(float))) # append land-sea mask as another channel onto inputfc

        preds_fc[[i],...] = cuda_to_numpy(STEPS_SCHEMES[stepscheme_fc](net_fc, inputfc)) # forecast time stepping, run the net_fc model using inputfc to produce the forecast and put in preds_fc.
        preds_fc[[i],...] = np.where(maskfc, 0, preds_fc[[i],...]) # fill up preds_fc where maskfc==0 (ocean)

    ## save preds with masked from lrvtest
    preds_fc_ma = np.ma.masked_array(preds_fc, np.tile(maskfc, (steps_autoreg,1,1,1))) # np.tile create a 4-D mask matrix, np.ma.masked_array mask the 4-d preds_fc

    print(f"forecast shape: {preds_fc_ma.shape}")

    ############################################
    ### Write prediction and true in netcdfs ###
    ############################################
    ## define output path
    outpath = f'./output/ENS-{teststart}/'
    if os.path.isdir(outpath) == False:
        os.mkdir(outpath)

    ## output low-res forecast fields as netcdf files
    outncfile = Dataset(outpath+'fcds_lrfc.nc', 'w', format='NETCDF4')
    outncfile.description = 'Low-res fields from Oceannet2.0 forecast'
    # dimensions
    outncfile.createDimension('lon', preds_fc_ma.shape[2])
    outncfile.createDimension('lat', preds_fc_ma.shape[1])
    outncfile.createDimension('ocean_time', preds_fc_ma.shape[0])
    outncfile.createDimension('channel', preds_fc_ma.shape[3])
    # variables
    nc_lon = outncfile.createVariable('lon', 'f4', ('lon',))
    nc_lat = outncfile.createVariable('lat', 'f4', ('lat',))
    nc_time = outncfile.createVariable('ocean_time', 'f8', ('ocean_time',))
    nc_chn = outncfile.createVariable('channel', 'f4', ('channel',))
    nc_field = outncfile.createVariable('pred', 'f8', ('channel','ocean_time','lat','lon',),fill_value=9.969209968386869e+36)
    # attributes
    nc_time.units = 'Days since 1858-11-17'
    nc_lon.units = 'degree_east'
    nc_lat.units = 'degree_north'
    nc_lon.axis = "X"
    nc_lat.axis = "Y"
    nc_chn.description='1:SSU, 2:SSV, 3:SSH, 4:SSKE'
    nc_field.coordinates="lon lat ocean_time channel"
    # data
    nc_chn[:] = np.asarray([1,2,3,4])
    nc_time[:] = datetime_fc
    nc_lat[:] = lrlats
    nc_lon[:] = lrlons
    nc_field[:] = np.transpose(preds_fc_ma,(3,0,1,2)) # shape: [channel,time,lat,lon]
    outncfile.close()