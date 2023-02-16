# Same as the notebook, to be run in the background.
# Jan 27, 2023. No maps are drawn.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob

def plot_background(ax):
    #ax.set_extent(extent_param)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    #ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 14}
    gl.ylabel_style = {'fontsize': 14}
    return ax

# Read the file with the location of the sediments

filename = 'DB_27122022.csv'
df_loc = pd.read_csv(filename,sep='\t')

# Set the path to SST files
path2sst = '/media/agostino/sailboat/neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.1/'
path2chla = '/media/agostino/sailboat/dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/v5.0-release/geographic/netcdf/chlor_a/monthly/v5.0/'

# Set the time window in which data are available at the monthly res.
str_start = '2007-01-01' #'YYYY-MM-DD'
str_end = '2013-12-01'
instant_start = np.datetime64(str_start)
instant_end = np.datetime64(str_end)

# Set the half width of the box over which to take the average.
dlat = 0.1
dlon = 0.1

allstats_db = np.array(['LONG','LAT','Water depth','Stations/Facies',
                        'SSTmin_K','SSTavg_K','SSTmax_K',
                        'SSTJan_K','SSTFeb_K','SSTMar_K','SSTApr_K','SSTMay_K','SSTJun_K',
                        'SSTJul_K','SSTAug_K','SSTSep_K','SSTOct_K','SSTNov_K','SSTDec_K',
                        'Chlamin_mg_m-3','Chlaavg_mg_m-3','Chlamax_mg_m-3',
                        'ChlaJan_mg_m-3','ChlaFeb_mg_m-3','ChlaMar_mg_m-3','ChlaApr_mg_m-3','ChlaMay_mg_m-3','ChlaJun_mg_m-3',
                        'ChlaJul_mg_m-3','ChlaAug_mg_m-3','ChlaSep_mg_m-3','ChlaOct_mg_m-3','ChlaNov_mg_m-3','ChlaDec_mg_m-3'
                       ])


for ss in range(len(df_loc['LONG'])): #np.arange(157,158): #range(len(df_loc['LONG'])): # Loop on the sites.
    # Select the point.
    lon0 = df_loc['LONG'][ss]
    lat0 = df_loc['LAT'][ss]
    wd0 = df_loc['Water depth'][ss]
    sf0 = str(df_loc['Stations/Facies'][ss])
    
    print('--------------------------------------------')
    print(sf0)
    print(wd0)
    print(sf0 + ' at ' + str(wd0) +' m')
    print(str(ss) +' out of '+ str(len(df_loc['LONG'])))
    print('--------------------------------------------')

    # Time loop
    instant = instant_start
    it_s_the_first_instant = True
    time_series = []
    sst_series = []
    chla_series = []
    sst_seasonal = np.zeros(12) # To compute the seasonal cycle
    sst_seasonal_count = np.zeros(12)
    chla_seasonal = np.zeros(12)
    chla_seasonal_count = np.zeros(12)
    nyears = 0 # Number of years.

    while instant <= instant_end:
        print(instant)
        pd_instant = pd.to_datetime(instant)
        year_oi = str(pd_instant.year).zfill(4)

        # Read the annual SST file.
        path_sst_oi = path2sst + '/' +  year_oi + '/'
        filename_sst = 'monthly_'+year_oi+'.nc'
        ds_sst = xr.open_dataset(path_sst_oi + filename_sst)

        # Set the path of the monthly Chla data for the current year.
        path_chla_oi = path2chla + '/' +  year_oi + '/'

        for mm in range(12):
            # Define the monthly time step in the time_series.
            month_oi = str(mm+1).zfill(2)
            time_series.extend([year_oi + '-' + month_oi])
            
            # Read the pointwise SST value.
            sst0 = ds_sst['analysed_sst'][mm].sel(lon=slice(lon0-dlon,lon0+dlon),
                                                  lat=slice(lat0-dlat,lat0+dlat)).mean(dim=['lon','lat'],skipna=True).values

            sst_series.extend([sst0.item()])
            sst_seasonal[mm] += sst0
            sst_seasonal_count[mm] += 1
            #print(sst_series)
            
            # Read the Chl-a data.
            filename_chla = 'ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx-'+year_oi+month_oi+'-fv5.0.nc'
            ds_chla = xr.open_dataset(path_chla_oi + filename_chla)

            chla0 = ds_chla['chlor_a'][0].sel(lon=slice(lon0-dlon,lon0+dlon),
                                              lat=slice(lat0+dlat,lat0-dlat)).mean(dim=['lon','lat'],skipna=True).values
            chla_series.extend([chla0.item()])
            chla_seasonal[mm] += chla0
            chla_seasonal_count[mm] += 1

            if it_s_the_first_instant:
                # Check the position of the point on a map.
                crs = ccrs.PlateCarree()
                lon_sst = ds_sst['lon'].values
                lat_sst = ds_sst['lat'].values
                l4_sst = ds_sst['analysed_sst'].values
                lon_chla = ds_chla['lon'].values
                lat_chla = ds_chla['lat'].values
                chla_map = ds_chla['chlor_a'].values

                it_s_the_first_instant = False
                
        instant = np.datetime64(str(int(year_oi)+1)+'-01-01') #+= np.timedelta64(1,'Y')
        nyears += 1

    # Compute the statistics of interest for SST and Chla.
    sst_t = np.array(sst_series)
    sst_min = np.nanmin(sst_t)
    sst_max = np.nanmax(sst_t)
    sst_avg = np.nanmean(sst_t)
    sst_seasonal_cycle = sst_seasonal/sst_seasonal_count 
    sst_stats = np.concatenate((np.array([lon0,lat0,wd0,sf0,sst_min,sst_avg,sst_max]),sst_seasonal_cycle))
    
    chla_t = np.array(chla_series)
    chla_min = np.nanmin(chla_t)
    chla_max = np.nanmax(chla_t)
    chla_avg = np.nanmean(chla_t)
    chla_seasonal_cycle = chla_seasonal/chla_seasonal_count 
    chla_stats = np.concatenate((np.array([chla_min,chla_avg,chla_max]),chla_seasonal_cycle))
    
    allstats_db = np.vstack([allstats_db,np.concatenate([sst_stats,chla_stats])])
    
    #print(allstats_db)
    
pd.DataFrame(allstats_db).to_csv('stats_prova.csv')
