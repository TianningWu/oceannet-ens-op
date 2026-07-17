import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import os
import cv2
import xarray as xr
import glob
from data_loaders_mercator import load_predict_grid_mask
from multiprocessing import Pool, cpu_count
from matplotlib import font_manager

PREFERRED_FONT = 'Open Sans'
FALLBACK_FONT = 'DejaVu Sans'


def set_default_font(size):
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_name = PREFERRED_FONT if PREFERRED_FONT in available_fonts else FALLBACK_FONT
    plt.rc('font', family='sans-serif', **{'sans-serif': [font_name], 'size': size})

def plt_coast_depth(ax,domainbnd):
    # load depth from high res dataset
    global lon2dhr_true, lat2dhr_true, hrh
    # Add coastline
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, 
                alpha=0.5, linestyle='--',linewidth=0.5, 
                xlocs=np.arange(-179,180,2), ylocs=np.arange(-89,89,2))
    ax.set_extent(domainbnd, crs=ccrs.PlateCarree())
    # Add bathymetry
    deplev = [100,150,250,500,1000,1500]
    ax.contour(lon2dhr_true, lat2dhr_true, hrh, deplev,
                transform=ccrs.PlateCarree(),colors='lightgray', linewidths=.5)
    return

def plot_ensmem(args,datain):
    # Set up the default font
    set_default_font(12)
    # plot each ensemble members speed and ssh
    # read data and parameters
    iday, ds, lon2d, lat2d, fc_dates = args
    ssh0,spd0,_,_,_,_ = datain
    # set up plot range and time
    init_timestr = np.datetime_as_string(fc_dates[0],unit='D')
    lonmin,lonmax,latmin,latmax = -94.7,-80,20.26,28.4
    figpath = './figs_ensmembers/'
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(23, 14))
    ## forecast
    # Create subplots for ssh
    for iens in range(40):
        idata = np.squeeze(ssh0[iens,:,:])
        ax = plt.subplot(6, 7, iens+1, projection=ccrs.PlateCarree())
        
        vmin,vmax,vint = -1.0, 1.01, 0.01
        cont_levels = np.arange(-1.,1.01,0.1)    
        # Plot contours
        im = ax.contourf(lon2d, lat2d, idata, 
                            transform=ccrs.PlateCarree(),
                            cmap=plt.cm.RdBu_r,
                            levels=np.arange(vmin,vmax,vint),extend='both')                      
        # Add zero contour line
        ax.contour(lon2d, lat2d, idata, cont_levels,
                    transform=ccrs.PlateCarree(),
                    colors='k', linewidths=0.8)
        # Add coastline
        plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
        # Add title
        ax.set_title(f'# {iens}')
        # Add colorbar
    cax = ax.inset_axes([1.1, 0.01, 0.07, 0.98])
    cbar = plt.colorbar(im, ax=ax, cax = cax, ticks=np.arange(vmin,vmax,0.25), format='%.2f',pad=0.02, shrink=1.)
    cbar.ax.set_title(f'[m]')

    valid_timestr = np.datetime_as_string(fc_dates[iday],unit='D')
    fig.suptitle(f'OceanNet2.0 120-day Ensemble Forecast SSH ({ssh0.shape[0]} members) \n Init: {init_timestr} Valid: {valid_timestr} Lead: {iday} days',horizontalalignment='center', fontsize=26)
    plt.tight_layout()
    plt.savefig(f'{figpath}/ssh_members_fconly_{iday}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(23, 14))
    
    # Create subplots for spd
    for iens in range(40):
        idata = np.squeeze(spd0[iens,:,:])
        ax = plt.subplot(6, 7, iens+1, projection=ccrs.PlateCarree())    
        vmin,vmax,vint = 0., 2.01, 0.01
        cont_levels = np.arange(0.,2.01,0.2)   
        # Plot contours
        im = ax.contourf(lon2d, lat2d, idata, 
                            transform=ccrs.PlateCarree(),
                            cmap=plt.cm.Reds,
                            levels=np.arange(vmin,vmax,vint),
                            extend='max')
        # Add spd contour line
        ax.contour(lon2d, lat2d, idata, cont_levels,
                    transform=ccrs.PlateCarree(),
                    colors='k', linewidths=0.8)
        # Add coastline
        plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
        # Add title
        ax.set_title(f'# {iens}')

    # Add colorbar
    cax = ax.inset_axes([1.1, 0.01, 0.07, 0.98])
    cbar = plt.colorbar(im, ax=ax, cax = cax, ticks=np.arange(vmin,vmax,0.25), format='%.2f',pad=0.02, shrink=1.)
    cbar.ax.set_title(f'[m/s]')
    # add figure title
    valid_timestr = np.datetime_as_string(fc_dates[iday],unit='D')
    fig.suptitle(f'OceanNet2.0 120-day Ensemble Forecast Sea Surface Speed ({ssh0.shape[0]} members) \n Init: {init_timestr} Valid: {valid_timestr} Lead: {iday} days',horizontalalignment='center', fontsize=26)
    # save fig
    plt.tight_layout()
    plt.savefig(f'{figpath}/spd_members_fconly_{iday}.png', dpi=150, bbox_inches='tight')
    plt.close()
    return

def plot_ensmean(args,datain):
    # Set up the default font
    set_default_font(24)
    # plot ensemble mean and spread, speed and ssh
    # read data and parameters
    iday, ds, lon2d, lat2d, fc_dates = args
    ssh0,spd0,ssh_avg,ssh_std,spd_avg,spd_std = datain
    # set up plot range and time
    init_timestr = np.datetime_as_string(fc_dates[0],unit='D')
    lonmin,lonmax,latmin,latmax = -94.7,-80,20.26,28.4
    figpath = './figs_ensmean/'

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(23, 14))

    # Create subplots for ssh
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    vmin,vmax,vint = -1.0, 1.01, 0.01
    cont_levels = np.arange(-1.,1.01,0.1)
    # Plot contours
    im = ax.contourf(lon2d, lat2d, ssh_avg, 
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.RdBu_r,
                        levels=np.arange(vmin,vmax,vint),extend='both')                      
    # Add zero contour line
    ax.contour(lon2d, lat2d, ssh_avg, cont_levels,
                transform=ccrs.PlateCarree(),
                colors='k', linewidths=0.8)
    # Add coastline
    plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
    # Add title
    ax.set_title(f'(a) SSH, ensemble mean', fontsize=24)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,0.2), format='%.1f',
                        pad=0.02, shrink=1.)
    cbar.ax.set_title(f'[m]',fontsize=20)

    # Create subplots for spd
    ax = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    vmin,vmax,vint = 0., 2.01, 0.01
    cont_levels = np.arange(0.,2.01,0.2)
    # Plot contours
    im = ax.contourf(lon2d, lat2d, spd_avg, 
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.Reds,
                        levels=np.arange(vmin,vmax,vint),
                        extend='max')
    # Add spd contour line
    ax.contour(lon2d, lat2d, spd_avg, cont_levels,
                transform=ccrs.PlateCarree(),
                colors='k', linewidths=0.8)
    # Add coastline
    plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
    # Add title
    ax.set_title(f'(c) Sea surface speed, ensemble mean', fontsize=24)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,0.2), format='%.1f',pad=0.02, shrink=1.)
    cbar.ax.set_title(f'[m/s]',fontsize=20)

    # Create subplots for ssh
    ax = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    vmin,vmax,vint = 0., 0.31, 0.01
    cont_levels = np.arange(-1.,1.01,0.1)
    # Plot contours
    im = ax.contourf(lon2d, lat2d, ssh_std, 
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.BuPu,
                        levels=np.arange(vmin,vmax,vint),extend='max')                      
    # Add zero contour line
    ax.contour(lon2d, lat2d, ssh_avg, cont_levels,
                transform=ccrs.PlateCarree(),
                colors='k', linewidths=0.8)
    # Add coastline
    plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
    # Add title
    ax.set_title(f'(b) SSH, ensemble standard deviation', fontsize=24)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,0.05), format='%.2f',
                        pad=0.02, shrink=1.)
    cbar.ax.set_title(f'[m]',fontsize=20)

    # Create subplots for spd
    ax = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    vmin,vmax,vint = 0., 0.51, 0.01
    cont_levels = np.arange(-2.,2.01,0.2)
    # Plot contours
    im = ax.contourf(lon2d, lat2d, spd_std, 
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.BuPu,
                        levels=np.arange(vmin,vmax,vint),
                        extend='max')
    # Add spd contour line
    ax.contour(lon2d, lat2d, spd_avg, cont_levels,
                transform=ccrs.PlateCarree(),
                colors='k', linewidths=0.8)
    # Add coastline
    plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
    # Add title
    ax.set_title(f'(d) Sea surface speed, ensemble standard deviation', fontsize=24)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,0.1), format='%.1f',pad=0.02, shrink=1.)
    cbar.ax.set_title(f'[m/s]',fontsize=20)
    # add figure title
    valid_timestr = np.datetime_as_string(fc_dates[iday],unit='D')
    fig.suptitle(f'OceanNet2.0 120-day Ensemble Forecast ({ssh0.shape[0]} members) \n Init: {init_timestr} Valid: {valid_timestr} Lead: {iday} days',horizontalalignment='center', fontsize=26)
    # save fig
    plt.tight_layout()
    plt.savefig(f'{figpath}/ssh_meanstd_fconly_{iday}.png', dpi=150, bbox_inches='tight')
    plt.close()
    return

def plot_features(args,datain):
    # Set up the default font
    set_default_font(24)
    # plot ensemble LC and Eddy features
    # read data and parameters
    iday, ds, lon2d, lat2d, fc_dates = args
    ssh0,spd0,ssh_avg,ssh_std,spd_avg,spd_std = datain
    # set up plot range and time
    init_timestr = np.datetime_as_string(fc_dates[0],unit='D')
    lonmin,lonmax,latmin,latmax = -94.7,-80,20.26,28.4
    figpath = './figs_features/'
    prob_spd0 = np.mean(spd0 > 0.77, axis=0)*100. # speed > 0.7m/s (1.5 knots), probability in %
    prob_ssh0 = np.mean(ssh0 > 0.17, axis=0)*100. # ssh > 17cm, probability in %
    prob_spd = np.where(prob_spd0<=0.01,np.nan,prob_spd0) # mask out 0 probability and land
    prob_ssh = np.where(prob_ssh0<=0.01,np.nan,prob_ssh0) # mask out 0 probability and land

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(23, 14))

    # Create subplots for ssh
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    vmin,vmax,vint = -1.0, 1.01, 0.01
    cont_levels = np.arange(-1.,1.01,0.1)
    # Plot contours
    im = ax.contourf(lon2d, lat2d, ssh_avg, 
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.RdBu_r,
                        levels=np.arange(vmin,vmax,vint),extend='both')                      
    # Add zero contour line
    ax.contour(lon2d, lat2d, ssh_avg, cont_levels,
                transform=ccrs.PlateCarree(),
                colors='k', linewidths=0.8)
    # Add coastline
    plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
    # Add title
    ax.set_title(f'(a) SSH, ensemble mean', fontsize=24)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,0.2), format='%.1f',pad=0.02, shrink=0.8)
    cbar.ax.set_title(f'[m]',fontsize=20)

    # Create subplots for spd
    ax = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    vmin,vmax,vint = 0., 2.01, 0.01
    cont_levels = np.arange(0.,2.01,0.2)
    # Plot contours
    im = ax.contourf(lon2d, lat2d, spd_avg, 
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.Reds,
                        levels=np.arange(vmin,vmax,vint),
                        extend='max')
    # Add spd contour line
    ax.contour(lon2d, lat2d, spd_avg, cont_levels,
                transform=ccrs.PlateCarree(),
                colors='k', linewidths=0.8)
    # Add coastline
    plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
    # Add title
    ax.set_title(f'(c) Sea surface speed, ensemble mean', fontsize=24)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,0.2), format='%.1f',pad=0.02, shrink=0.8)
    cbar.ax.set_title(f'[m/s]',fontsize=20)

    # Create subplots for ssh
    ax = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    vmin,vmax,vint = 0.0, 100.1, 1.
    cont_levels = [0.17]
    # Plot contours pf every ens member
    for iens in range(ssh0.shape[0]):
        ax.contour(lon2d, lat2d, ssh0[iens,:,:], cont_levels,
                    transform=ccrs.PlateCarree(),colors='k', linewidths=0.5)
    # Add zero contour line
    ax.contour(lon2d, lat2d, ssh_avg, cont_levels,
                transform=ccrs.PlateCarree(),
                colors='r', linewidths=1.)
    # plot probility of LC
    im = ax.contourf(lon2d, lat2d, prob_ssh, 
                    transform=ccrs.PlateCarree(),cmap=plt.cm.Blues,
                    levels=np.arange(vmin,vmax,vint),extend='neither')                      
    # Add coastline
    plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
    # Add title
    ax.set_title(f'(b) Probability of Loop Current and Eddy (SSH > 17 cm)', fontsize=24)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,20.), format='%.0f',pad=0.02, shrink=0.8)
    cbar.ax.set_title(f'[%]',fontsize=20)

    # Create subplots for spd
    ax = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    vmin,vmax,vint = 0.0, 100.1, 1.
    cont_levels = [60.]
    # Plot contours
    im = ax.contourf(lon2d, lat2d, prob_spd, 
                    transform=ccrs.PlateCarree(),cmap=plt.cm.Reds,
                    levels=np.arange(vmin,vmax,vint),extend='neither')
    # Add spd contour line
    ax.contour(lon2d, lat2d, prob_spd, cont_levels,
                transform=ccrs.PlateCarree(),
                colors='k', linewidths=1.5)
    # Add coastline
    plt_coast_depth(ax,[lonmin,lonmax,latmin,latmax])
    # Add title
    ax.set_title(f'(d) Probability of Loop Current speed (speed > 1.5 kn)', fontsize=24)
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,20.), format='%.0f',pad=0.02, shrink=0.8)
    cbar.ax.set_title(f'[%]',fontsize=20)
    # add figure title
    valid_timestr = np.datetime_as_string(fc_dates[iday],unit='D')
    fig.suptitle(f'OceanNet2.0 120-day Ensemble Forecast ({ssh0.shape[0]} members) \n Init: {init_timestr} Valid: {valid_timestr} Lead: {iday} days',horizontalalignment='center', fontsize=26)
    # save fig
    plt.tight_layout()
    plt.savefig(f'{figpath}/ssh_feat_fconly_{iday}.png', dpi=150, bbox_inches='tight')
    plt.close()
    return

def plotframe(args):
    # load paramenters
    iday, ds, lon2d, lat2d, fc_dates = args
    # Open the dataset for the specific day
    # load ensemble predictions
    ssh0 = ds.variables['pred'][:,:,:,iday,2]  # shape: (ens,lon,lat,time,channel)
    ssu0 = ds.variables['pred'][:,:,:,iday,0]  # shape: (ens,lon,lat,time,channel)
    ssv0 = ds.variables['pred'][:,:,:,iday,1]  # shape: (ens,lon,lat,time,channel)
    spd0 = np.sqrt(ssu0**2 + ssv0**2)
    # averaging and standard deviation
    ssh_avg = ssh0.mean(axis=0)
    ssh_std = ssh0.std(axis=0)
    spd_avg = spd0.mean(axis=0)
    spd_std = spd0.std(axis=0)
    # put all data into a argument to pass into functions
    datain = (ssh0,spd0,ssh_avg,ssh_std,spd_avg,spd_std)
    # plot ens mean
    plot_ensmean(args,datain)
    # plot ens features
    plot_features(args,datain)
    # plot ens members
    plot_ensmem(args,datain)
    print(np.datetime_as_string(fc_dates[iday],unit='D'))
    return

if __name__ == '__main__':
    # load depth from high res dataset
    _, hrlons, hrlats, hrh = load_predict_grid_mask() # load the high res depth and grid
    lat2dhr_true, lon2dhr_true = np.meshgrid(hrlats,hrlons)
    hrh = np.transpose(hrh,(1,0))

    # Path to your NetCDF files (adjust the path and file pattern)
    file_list = './output/fcds_lrfc_ens.nc'
    # Open multiple files at once
    ds = xr.open_dataset(file_list).transpose('ens',"lon", "lat", "ocean_time",'channel')
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    fc_dates = ds.variables['ocean_time'][:]
    # Create 2D coordinate grids
    lat2d, lon2d = np.meshgrid(lats,lons)

    log_parallel=1 # logic swith to turn on/off parallel plot (0=sequential, 1=parallel)
    animfmt='mp4' #'gif' or 'mp4'
    # Prepare arguments for parallel processing
    args = [(iday, ds, lon2d, lat2d, fc_dates) for iday in range(len(fc_dates))]
    if (log_parallel == 0):
        for iarg in args:
            plotframe(iarg)
    elif (log_parallel == 1):
        num_processes = min(cpu_count(), len(args))  # Use number of available cores or number of frames, whichever is smaller
        # Create a pool of workers and process frames in parallel
        with Pool(processes=num_processes) as ipool:
            ipool.map(plotframe, args)
    # make video from the figures
    init_timestr = np.datetime_as_string(fc_dates[0],unit='D').replace('-', '')
    if (animfmt=='mp4'):
        figpath = './figs_ensmembers/'
        outpath = './archives/ens_mem/'
        os.system(f'module load FFmpeg/5.1.2-GCCcore-12.2.0; ffmpeg -framerate 2 -y -i {figpath}/ssh_members_fconly_%d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {outpath}/oceannet_sshmembers_{init_timestr}.mp4')
        os.system(f'module load FFmpeg/5.1.2-GCCcore-12.2.0; ffmpeg -framerate 2 -y -i {figpath}/spd_members_fconly_%d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {outpath}/oceannet_spdmembers_{init_timestr}.mp4')
        figpath = './figs_features/'
        outpath = './archives/ens_features/'
        os.system(f'module load FFmpeg/5.1.2-GCCcore-12.2.0; ffmpeg -framerate 2 -y -i {figpath}/ssh_feat_fconly_%d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {outpath}/oceannet_ensfeatures_{init_timestr}.mp4')
        figpath = './figs_ensmean/'
        outpath = './archives/ens_mean/'
        os.system(f'module load FFmpeg/5.1.2-GCCcore-12.2.0; ffmpeg -framerate 2 -y -i {figpath}/ssh_meanstd_fconly_%d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {outpath}/oceannet_ensmean_{init_timestr}.mp4')
    elif (animfmt=='gif'):
        figpath = './figs_ensmembers/'
        outpath = './archives/ens_mem/'
        os.system(f'module load FFmpeg/5.1.2-GCCcore-12.2.0; ffmpeg -framerate 2 -y -i {figpath}/ssh_members_fconly_%d.png -filter_complex "scale=2048:-1:flags=lanczos,pad=iw:ih:(ow-iw)/2:(oh-ih)/2:color=white,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" {outpath}/oceannet_sshmembers_{init_timestr}.gif')
        os.system(f'module load FFmpeg/5.1.2-GCCcore-12.2.0; ffmpeg -framerate 2 -y -i {figpath}/spd_members_fconly_%d.png -filter_complex "scale=2048:-1:flags=lanczos,pad=iw:ih:(ow-iw)/2:(oh-ih)/2:color=white,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" {outpath}/oceannet_spdmembers_{init_timestr}.gif')
        figpath = './figs_features/'
        outpath = './archives/ens_features/'
        os.system(f'module load FFmpeg/5.1.2-GCCcore-12.2.0; ffmpeg -framerate 2 -y -i {figpath}/ssh_feat_fconly_%d.png -filter_complex "scale=2048:-1:flags=lanczos,pad=iw:ih:(ow-iw)/2:(oh-ih)/2:color=white,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" {outpath}/oceannet_ensfeatures_{init_timestr}.gif')
        figpath = './figs_ensmean/'
        outpath = './archives/ens_mean/'
        os.system(f'module load FFmpeg/5.1.2-GCCcore-12.2.0; ffmpeg -framerate 2 -y -i {figpath}/ssh_meanstd_fconly_%d.png -filter_complex "scale=2048:-1:flags=lanczos,pad=iw:ih:(ow-iw)/2:(oh-ih)/2:color=white,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" {outpath}/oceannet_ensmean_{init_timestr}.gif')
    ds.close()
