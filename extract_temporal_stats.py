# Same as the notebook, to be run in the background.
# Jan 27, 2023. No maps are drawn.
# March 2023. Compute the energy at depth with the radiation absorption coeff.
# April 4, 2023. Use the DB_28032023_nofirst_col.csv file as an input.

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

filename = 'DB_28032023_nofirstcol.csv' #'DB_27122022.csv'
df_loc = pd.read_csv(filename,sep='\t')

# Set the path to SST files
path2sst = '/media/agostino/sailboat/neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.1/'
path2chla = '/media/agostino/sailboat/dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/v5.0-release/geographic/netcdf/chlor_a/monthly/v5.0/'
path2kd = '/media/agostino/sailboat/dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/v5.0-release/geographic/netcdf/kd/monthly/v5.0/'
path2sw = '/media/agostino/sailboat/satproj_klima_dwd/ESA_CLoud_CCI/CLD_PRODUCTS/v3.0/L3C/AVHRR-AM/AVHRR_METOPA/'

# Set the time window in which data are available at the monthly res.
str_start = '2008-01-01' #'YYYY-MM-DD'
str_end = '2013-12-01'
instant_start = np.datetime64(str_start)
instant_end = np.datetime64(str_end)

# Set the half width of the box over which to take the average.
dlat = 0.1
dlon = 0.1
dlonn = 0.2
dlatt = 0.2
dlonnn = 0.5
dlattt = 0.5

allstats_db = np.array(['LONG','LAT','Water depth','Stations/Facies',
                        'SSTmin_K','SSTavg_K','SSTmax_K',
                        'SSTJan_K','SSTFeb_K','SSTMar_K','SSTApr_K','SSTMay_K','SSTJun_K',
                        'SSTJul_K','SSTAug_K','SSTSep_K','SSTOct_K','SSTNov_K','SSTDec_K',
                        'Chlamin_mg_m-3','Chlaavg_mg_m-3','Chlamax_mg_m-3',
                        'ChlaJan_mg_m-3','ChlaFeb_mg_m-3','ChlaMar_mg_m-3','ChlaApr_mg_m-3','ChlaMay_mg_m-3','ChlaJun_mg_m-3',
                        'ChlaJul_mg_m-3','ChlaAug_mg_m-3','ChlaSep_mg_m-3','ChlaOct_mg_m-3','ChlaNov_mg_m-3','ChlaDec_mg_m-3'
                       ])

list_of_nans_sst = []
list_of_nans_chla = []

for ss in range(len(df_loc['LONG'])): # np.arange(498,500): # range(len(df_loc['LONG'])): # Loop on the sites.
    # Select the point.
    lon0 = df_loc['LONG'][ss]
    lat0 = df_loc['LAT'][ss]
    wd0 = df_loc['Water depth'][ss]
    sf0 = str(df_loc['Stations/Facies'][ss])
    
    print('--------------------------------------------')
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

    the_area_has_been_extended_sst = False
    the_area_has_been_extended_chla = False
    
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
            if np.isnan(sst0): # Extend the area over which the sst is read.
                sst0 = ds_sst['analysed_sst'][mm].sel(lon=slice(lon0-dlonn,lon0+dlonn),
                                                      lat=slice(lat0-dlatt,lat0+dlatt)).mean(dim=['lon','lat'],skipna=True).values
                the_area_has_been_extended_sst = True
                
            sst_series.extend([sst0.item()])
            sst_seasonal[mm] += sst0
            sst_seasonal_count[mm] += 1
            #print(sst_series)
            
            # Read the Chl-a data.
            filename_chla = 'ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx-'+year_oi+month_oi+'-fv5.0.nc'
            ds_chla = xr.open_dataset(path_chla_oi + filename_chla)

            chla0 = ds_chla['chlor_a'][0].sel(lon=slice(lon0-dlon,lon0+dlon),
                                              lat=slice(lat0+dlat,lat0-dlat)).mean(dim=['lon','lat'],skipna=True).values
            if np.isnan(chla0): # Extend the area over which the sst is read.
                chla0 = ds_chla['chlor_a'][0].sel(lon=slice(lon0-dlonn,lon0+dlonn),
                                                 lat=slice(lat0-dlatt,lat0+dlatt)).mean(dim=['lon','lat'],skipna=True).values
                the_area_has_been_extended_chla = True

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
    
    if the_area_has_been_extended_sst:
        list_of_nans_sst.append(sf0+'_at_'+str(wd0))
        print('sst:',list_of_nans_sst)    
    if the_area_has_been_extended_chla:
        list_of_nans_chla.append(sf0+'_at_'+str(wd0))
        print('chla:',list_of_nans_chla)

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
pd.DataFrame(list_of_nans_sst).to_csv('list_sites_larger_area_sst.csv')
pd.DataFrame(list_of_nans_chla).to_csv('list_sites_larger_area_chla.csv')

# Analysis on KD and SW. SW is the shortwave flux at the surface and SWz is at depth, computed as
# SWz = SW*exp(-KD*wd0)

allstats_d2 = np.array(['LONG','LAT','Water depth','Stations/Facies',
                        'KDmin_m-1','KDavg_m-1','KDmax_m-1',
                        'KDJan_m-1','KDFeb_m-1','KDMar_m-1','KDApr_m-1','KDMay_m-1','KDJun_m-1',
                        'KDJul_m-1','KDAug_m-1','KDSep_m-1','KDOct_m-1','KDNov_m-1','KDDec_m-1',
                        'SWmin_W_m-2','SWavg_W_m-2','SWmax_W_m-2',
                        'SWJan_W_m-2','SWFeb_W_m-2','SWMar_W_m-2','SWApr_W_m-2','SWMay_W_m-2','SWJun_W_m-2',
                        'SWJul_W_m-2','SWAug_W_m-2','SWSep_W_m-2','SWOct_W_m-2','SWNov_W_m-2','SWDec_W_m-2',
                        'SWmin_W_m-2','SWavg_W_m-2','SWmax_W_m-2',
                        'SWJan_W_m-2','SWFeb_W_m-2','SWMar_W_m-2','SWApr_W_m-2','SWMay_W_m-2','SWJun_W_m-2',
                        'SWJul_W_m-2','SWAug_W_m-2','SWSep_W_m-2','SWOct_W_m-2','SWNov_W_m-2','SWDec_W_m-2'
                       ])

list_of_nans_kd = []

for ss in range(len(df_loc['LONG'])): #range(len(df_loc['LONG'])): # Loop on the sites.
    # Select the point.
    lon0 = df_loc['LONG'][ss]
    lat0 = df_loc['LAT'][ss]
    wd0 = df_loc['Water depth'][ss]
    sf0 = str(df_loc['Stations/Facies'][ss])
    
    print('--------------------------------------------')
    print(sf0 + ' at ' + str(wd0) +' m')
    print(str(ss) +' out of '+ str(len(df_loc['LONG'])))
    print('--------------------------------------------')

    # Time loop
    instant = instant_start
    it_s_the_first_instant = True
    time_series = []
    kd_series = []
    sw_series = []
    swz_series = []
    kd_seasonal = np.zeros(12) # To compute the seasonal cycle
    kd_seasonal_count = np.zeros(12)
    sw_seasonal = np.zeros(12)
    sw_seasonal_count = np.zeros(12)
    swz_seasonal = np.zeros(12)
    swz_seasonal_count = np.zeros(12)
    nyears = 0 # Number of years.

    the_area_has_been_extended_kd = False
    the_area_has_been_extended_sw = False

    while instant <= instant_end:
        print(instant)
        pd_instant = pd.to_datetime(instant)
        year_oi = str(pd_instant.year).zfill(4)

        # Set the path of the monthly Chla data for the current year.
        path_kd_oi = path2kd + '/' +  year_oi + '/'
        path_sw_oi = path2sw + '/' +  year_oi + '/'

        for mm in range(12):
            # Define the monthly time step in the time_series.
            month_oi = str(mm+1).zfill(2)
            time_series.extend([year_oi + '-' + month_oi])
                        
            # Read the KD data.
            filename_kd = 'ESACCI-OC-L3S-K_490-MERGED-1M_MONTHLY_4km_GEO_PML_KD490_Lee-'+year_oi+month_oi+'-fv5.0.nc'
            ds_kd = xr.open_dataset(path_kd_oi + filename_kd)

            kd0 = ds_kd['kd_490'][0].sel(lon=slice(lon0-dlon,lon0+dlon),
                                         lat=slice(lat0+dlat,lat0-dlat)).mean(dim=['lon','lat'],skipna=True).values
            if np.isnan(kd0): # Extend the area over which the sst is read.
                dlonn = 0.2
                dlatt = 0.2
                kd0 = ds_kd['kd_490'][0].sel(lon=slice(lon0-dlonn,lon0+dlonn),
                                             lat=slice(lat0-dlatt,lat0+dlatt)).mean(dim=['lon','lat'],skipna=True).values
                the_area_has_been_extended_kd = True

            kd_series.extend([kd0.item()])
            kd_seasonal[mm] += kd0
            kd_seasonal_count[mm] += 1

            # Read the SW data. 
            # There should not be NaN, as the data are gap-free and are defined over land, as well.
            # Both fluxes are positive and the upward is smaller than the downward.
            filename_sw = year_oi+month_oi+'-ESACCI-L3C_CLOUD-CLD_PRODUCTS-AVHRR_METOPA-fv3.0.nc'
            ds_sw = xr.open_dataset(path_sw_oi + filename_sw)

            swup0 = ds_sw['boa_swup'][0].sel(lon=slice(lon0-dlonnn,lon0+dlonnn),
                                             lat=slice(lat0-dlattt,lat0+dlattt)).mean(dim=['lon','lat'],skipna=True).values
            swdn0 = ds_sw['boa_swdn'][0].sel(lon=slice(lon0-dlonnn,lon0+dlonnn),
                                             lat=slice(lat0-dlattt,lat0+dlattt)).mean(dim=['lon','lat'],skipna=True).values
            #print('swup',swup0,'swdn',swdn0)
            
            # Compute the net shortwave flux.
            sw0 = swdn0-swup0
                
            sw_series.extend([sw0.item()])
            sw_seasonal[mm] += sw0
            sw_seasonal_count[mm] += 1
            
            # Approximately compute the net shortwave flux at depth.
            swz0 = sw0*np.exp(-kd0*wd0)
            
            swz_series.extend([swz0.item()])
            swz_seasonal[mm] += swz0
            swz_seasonal_count[mm] += 1
            
            if it_s_the_first_instant:
                # Check the position of the point on a map.
                crs = ccrs.PlateCarree()
                lon_kd = ds_kd['lon'].values
                lat_kd = ds_kd['lat'].values
                kd_map = ds_kd['kd_490'].values
                lon_sw = ds_sw['lon'].values
                lat_sw = ds_sw['lat'].values
                swup_map = ds_sw['boa_swup'].values
                swdn_map = ds_sw['boa_swdn'].values

                it_s_the_first_instant = False
                
        instant = np.datetime64(str(int(year_oi)+1)+'-01-01') #+= np.timedelta64(1,'Y')
        nyears += 1
        
    if the_area_has_been_extended_kd:
        list_of_nans_kd.append(sf0+'_at_'+str(wd0))
        print('kd:',list_of_nans_kd)    

    # Compute the statistics of interest for SST and Chla.
    kd_t = np.array(kd_series)
    kd_min = np.nanmin(kd_t)
    kd_max = np.nanmax(kd_t)
    kd_avg = np.nanmean(kd_t)
    kd_seasonal_cycle = kd_seasonal/kd_seasonal_count 
    kd_stats = np.concatenate((np.array([lon0,lat0,wd0,sf0,kd_min,kd_avg,kd_max]),kd_seasonal_cycle))
    
    sw_t = np.array(sw_series)
    sw_min = np.nanmin(sw_t)
    sw_max = np.nanmax(sw_t)
    sw_avg = np.nanmean(sw_t)
    sw_seasonal_cycle = sw_seasonal/sw_seasonal_count 
    sw_stats = np.concatenate((np.array([sw_min,sw_avg,sw_max]),sw_seasonal_cycle))

    swz_t = np.array(swz_series)
    swz_min = np.nanmin(swz_t)
    swz_max = np.nanmax(swz_t)
    swz_avg = np.nanmean(swz_t)
    swz_seasonal_cycle = swz_seasonal/swz_seasonal_count 
    swz_stats = np.concatenate((np.array([swz_min,swz_avg,swz_max]),swz_seasonal_cycle))

    allstats_d2 = np.vstack([allstats_d2,np.concatenate([kd_stats,sw_stats,swz_stats])])
    
    #print(allstats_db)
    
pd.DataFrame(allstats_d2).to_csv('stats_prov2.csv')
pd.DataFrame(list_of_nans_kd).to_csv('list_sites_larger_area_kd.csv')
