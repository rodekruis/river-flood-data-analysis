
import xarray as xr 
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from rasterstats import zonal_stats
import rasterio
from rasterio.features import geometry_mask
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
from shapely import wkt
import rioxarray as rio

# checkVectorFormat(cfg.admPath)

def zonalStats(rasterDA, vectorGDF, IDhead, measure='max'):
    '''
    default measure is max 
    '''
    if vectorGDF.crs != rasterDA.rio.crs:
        vectorGDF.to_crs(rasterDA.rio.crs, inplace=True)

    geometries = vectorGDF['geometry']
    mask = geometry_mask(geometries, transform=rasterDA.rio.transform(), invert=True, out_shape=rasterDA.shape)
    masked_data = np.where(mask, rasterDA, np.nan)
    affine = rasterDA.rio.transform()

    zs = zonal_stats(
        geometries,
        masked_data,
        affine=affine,
        stats=[f'{measure}'],
        all_touched=True, 
        nodata=-9999
        )
    max_values = [zone[f'{measure}'] for zone in zs]

    communeMax_df = pd.DataFrame({
        f'{IDhead}': vectorGDF[f'{IDhead}'],
        f'{measure}': max_values
    })
    communeMax_gdf = gpd.GeoDataFrame(communeMax_df, geometry=vectorGDF.geometry)
    return communeMax_gdf


def query(rasterDA, pointGDF):
    '''
    pointGDF: GeoDataFrame with geometry describing the locations of the different points, nothing else
    rasterDA: xarray.DataArray with raster data
    '''
    coordinates = pointGDF.get_coordinates().values  # 2D array with latitudes and longitudes
    point_latitude = coordinates[:, 1]  # Extract all latitudes (first column) # REVERSE for now, leads to rsults 
    point_longitude = coordinates[:, 0]  # Extract all longitudes (second column) # Reverse becaise od l
    
    # Query the raster data for each point and store the result in a new 'rastervalue' column
    pointGDF['rastervalue'] = [
        rasterDA.sel(latitude=lat, longitude=lon, method='nearest').values.item()  # Get scalar value
        for lat, lon in zip(point_latitude, point_longitude)
        ]
    # print (pointGDF['rastervalue'])
    return pointGDF


def aggregation (rasterDA, vector, method, nrEnsemble=None, timestamp=None, measure=None, IDhead=None): 
    '''
    Decision tree for data reduction by aggregation.
    
    Parameters:
    rasterDA : xarray.DataArray
        The raster to aggregate on. Should be a preloaded DataArray with spatial tags.
    vectorGDF : geopandas.GeoDataFrame or path or string
        Vector data for aggregation or querying. Should contain points for 'point' or 'bufferpoint' methods,
        or polygons for the 'adm_unit' method (entailing the administrative units).
    method : str
        Aggregation method, one of ['point', 'polygon'] (and 'bufferpoint' if implemented).
    nrEnsemble : int, optional
        Ensemble number to select in case of multi-dimensional raster data.
    timestamp : str or datetime, optional
        Timestamp to select in case of time-dependent raster data.
    measure : str, optional
        Statistical measure to calculate for zonal statistics (e.g., 'mean', 'sum'). Only relevant for polygon or buffer aggregation

    Returns:
    GeoDataFrame 
    '''
    
    # reducing dimensions to two: one raster for the ensemble number and timestamp, depending on whether they've already been singled out
    if 'number' in rasterDA.dims or 'time' in rasterDA.dims:
        # If they are present, reduce to a single ensemble and timestamp
        if nrEnsemble is not None and 'number' in rasterDA.dims:
            rasterDA = rasterDA.sel(number=nrEnsemble)
        if timestamp is not None and 'time' in rasterDA.dims:
            rasterDA = rasterDA.sel(time=timestamp)
    # aggregation is about the model's location in relation to the model's output raster, therefore placement is in the model's framework
        # this argument is ignored when considering polygons, because on that level precision doesnt really matter
    vectorGDF = checkVectorFormat(vector, shapeType=method, crs=rasterDA.rio.crs, placement='real')
    if method == 'point': 
        pointgdf = query(rasterDA, vectorGDF)
        return pointgdf 
    # elif method == 'bufferpoint': 
    #     return pointgdf
    elif method == 'polygon': 
        admgdf = zonalStats(rasterDA, vectorGDF, IDhead, measure)
        return admgdf
    else: 
        raise ValueError ("Invalid method, choose 'point', or 'adm_unit'")


