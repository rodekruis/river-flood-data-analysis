import xarray as xr 
import rioxarray as rio 

def openThreshold(DataDir, crs, RPyr, area, Q_da_forecast ):
    threshold_ds = xr.load_dataset(DataDir / f"flood_threshold_glofas_v4_rl_{RPyr:.1f}.nc")
    threshold_ds.rio.write_crs(crs, inplace=True)

    threshold_da = threshold_ds[f"rl_{RPyr:.1f}"].sel(
        lat=slice(area[0], area[2]),
        lon=slice(area[1], area[3])
    )
    threshold_da = threshold_da.rename({'lat': 'latitude', 'lon': 'longitude'})
     #first getting just one Q_da of the first year to get the shape right
    threshold_da = threshold_da.interp(latitude=Q_da_forecast.latitude, longitude=Q_da_forecast.longitude)
    return threshold_da