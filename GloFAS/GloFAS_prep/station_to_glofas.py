from comparison.pointMatching import find_corresponding_point_within_box
import pandas as pd 
import xarray as xr 
import GloFAS.GloFAS_prep.configuration as cfg
area = cfg.MaliArea
# Load data
stations_ups_csv = f'{cfg.DataDir}/DNHMali_2019/Stations_ups_area_DNH.csv'
upstream_area_nc = f"{cfg.DataDir}/auxiliary/uparea_glofas_v4_0.nc"

stations_upsarea_df = pd.read_csv(stations_ups_csv)
print (stations_upsarea_df.head)
glofas_ups_area_ds = xr.open_dataset(upstream_area_nc, engine="h5netcdf")
#glofas_ups_area_ds["longitude"] = glofas_ups_area_ds["longitude"].where(glofas_ups_area_ds["longitude"] < 180, glofas_ups_area_ds["longitude"] - 360)
glofas_ups_area_da = glofas_ups_area_ds['uparea'] #.sel(
    #     latitude=slice(area[0], area[2]),
    #     longitude=slice(area[1], area[3])
    # )  # Access the upstream area variable
# glofas_ups_area_da.plot()

# Apply the function row-wise
for i, row in stations_upsarea_df.iterrows():
    glofas_x, glofas_y, area_diff = find_corresponding_point_within_box(
        station_lon=row['Lon'],
        station_lat=row['Lat'],
        ups_area_point_m=row['Catchment area (km2)'] * 1e6,  # Convert from km² to m²
        stationName=row['Station names'],
        glofas_ups_area_xr=glofas_ups_area_da,
        radius_m=10000
    )
    stations_upsarea_df.loc[i, 'Glofas_Point_X'] = glofas_x
    stations_upsarea_df.loc[i, 'Glofas_Point_Y'] = glofas_y
    stations_upsarea_df.loc[i, 'Area_Diff'] = area_diff

# Print or save the updated DataFrame
print(stations_upsarea_df.head())
stations_upsarea_df.to_csv(f'{cfg.DataDir}/stations/stations_Glofas_coordinates.csv')
