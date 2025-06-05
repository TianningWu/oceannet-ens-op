import numpy as np
import os
import xarray as xr

if __name__ == '__main__':
    # Path to your NetCDF files (adjust the path and file pattern)
    file_list = './output/fcds_lrfc_ens.nc'
    animfmt='gif'
    # Open multiple files at once
    ds = xr.open_dataset(file_list).transpose('ens',"lon", "lat", "ocean_time",'channel')
    fc_dates = ds.variables['ocean_time'][:]
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