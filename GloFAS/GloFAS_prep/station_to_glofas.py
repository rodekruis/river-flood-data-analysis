from comparison.pointMatching import find_closestcorresponding_point
import pandas as pd 
import xarray as xr 
import GloFAS.GloFAS_prep.configuration as cfg

# Load data
stations_ups_csv = f'{cfg.DataDir}/Stations_ups_area_DNH.csv'
upstream_area_nc = f"{cfg.DataDir}/uparea_glofas_v4_0.nc"

stations_upsarea_df = pd.read_csv(stations_ups_csv)

glofas_ups_area_ds = xr.open_dataset(upstream_area_nc, engine="h5netcdf")
glofas_ups_area_da = glofas_ups_area_ds['uparea']  # Access the upstream area variable

# A wrapper function for apply
def find_closest_wrapper(row):
    return find_closestcorresponding_point(
        point_x=row['Lon'],
        point_y=row['Lat'],
        ups_area_point_m=row['Catchment area (km2)'] * 1e6,  # Convert from km² to m²
        glofas_ups_area_xr=glofas_ups_area_da,
        radius_m=6000
    )

# Apply the function row-wise
results = stations_upsarea_df.apply(find_closest_wrapper, axis=1)

# Extract results into separate columns
stations_upsarea_df['Glofas_Point_X'], stations_upsarea_df['Glofas_Point_Y'], stations_upsarea_df['Area_Diff'] = zip(*results)


# Print or save the updated DataFrame
print(stations_upsarea_df.head())
stations_upsarea_df.to_csv(f'{cfg.DataDir}/stations_Glofas_coordinates.csv')
