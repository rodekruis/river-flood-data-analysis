
import xarray as xr 
import geopandas as gpd
import pandas as pd
from pathlib import Path
from rasterstats import zonal_stats
from shapely import wkt
import rioxarray as rio

def checkVectorFormat(vector, shapeType=None, crs='EPSG:4326', placement='real'):
    ''' 
    Transforms various forms of vector inputs into a proper GeoDataFrame.

    Parameters:
    vector : str, Path or already GeoDataFrame
        Path to the vector file, which can be a shapefile (.shp), GeoPackage (.gpkg), or CSV (.csv).
        Or potentially, already a geodataframe
    shapeType : str, optional
        Indicates the type of geometry that needs to be represented, either 'point' or 'polygon' (only relevant if the file is a CSV) 
    crs : str, default 'EPSG:4326'
        Coordinate reference system to assign to the resulting GeoDataFrame.
    placement: Str, default is 'real' 
        String describing whether placement of station location is in real life or in model 
        This is relevant because sometimes points (like discharge stations) can have two locations: 
            one in 'real' life coordinate system, one in the 'model' coordinate system
    Returns:
    GeoDataFrame
        A GeoDataFrame with the specified CRS, either point or polygon type, depending on input.
    '''
    
    # If the input is a file path
    if isinstance(vector, (str, Path)):
        vector = str(vector)  # Ensure compatibility with string operations
        # Check file extensions and load accordingly
        if vector.endswith(('.shp', '.gpkg')):
            vectorGDF = gpd.read_file(vector).to_crs(crs)
        
        elif vector.endswith('.csv'):
            # Load CSV and convert to GeoDataFrame based on shapeType
            df = pd.read_csv(vector)
            if shapeType=='point':
                if placement=='real': # default
                    # Some options describing columns describing placement in COORDINATES: ADD MORE in case your column name is not there
                    if {'Longitude', 'Latitude'}.issubset(df.columns):
                        vectorGDF = gpd.GeoDataFrame(
                            df,
                            geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
                            crs=crs
                            )
                    elif {'x', 'y'}.issubset(df.columns):
                        vectorGDF = gpd.GeoDataFrame(
                            df,
                            geometry=gpd.points_from_xy(df['x'], df['y']),
                            crs=crs
                            ) 
                    elif {'StationLon', 'StationLat'}.issubset(df.columns):
                        vectorGDF = gpd.GeoDataFrame(
                            df,
                            geometry=gpd.points_from_xy(df['StationLon'], df['StationLat']),
                            crs=crs
                            )
                    elif {'longitude', 'latitude'}.issubset(df.columns):
                        vectorGDF = gpd.GeoDataFrame(
                            df,
                            geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
                            crs=crs
                            )
                    else: 
                        raise ValueError ("Column names corresponding to longitude latitude cannot be found (maybe you would like to add more? :)) ")
                elif placement=='model': 
                    if {'LisfloodX', 'LisfloodY'}.issubset(df.columns): # this is the vector file 
                        vectorGDF = gpd.GeoDataFrame(
                            df,
                            geometry=gpd.points_from_xy(df['LisfloodX'], df['LisfloodY']),
                            crs=crs
                            )
                    else: 
                        raise ValueError ("Column names for the model locator ('LisfloodX, LisfloodY') cannot be found")
                else: 
                    raise ValueError("describe whether you want real or model vector")
            elif shapeType == 'polygon':
                if 'geometry' not in df.columns:
                    raise ValueError("CSV must contain a 'geometry' column with WKT format for polygon data.")
                df['geometry'] = df['geometry'].apply(wkt.loads)
                vectorGDF = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
            
            else:
                raise ValueError("When loading from a CSV, shapeType must be either 'point' or 'polygon'.")
        
        else:
            raise ValueError("File must be of type '.shp', '.gpkg', or '.csv'.")

    # If vector is already a GeoDataFrame
    elif isinstance(vector, gpd.GeoDataFrame):
        vectorGDF = vector.to_crs(crs) if vector.crs != crs else vector

    else:
        raise TypeError("The input vector must be a file path (str or Path) or a GeoDataFrame.")
    
    
    return vectorGDF

# checkVectorFormat(cfg.admPath)

def zonalStats(rasterDA, vectorGDF, measure='max'):
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
        f'ADM{self.adminLevel}': vectorGDF[f'ADM{self.adminLevel}_FR'],
        'max': max_values
    })
    communeMax_gdf = gpd.GeoDataFrame(communeMax_df, geometry=vectorGDF.geometry)
    return communeMax_gdf

def query(rasterDA, pointGDF, nrEnsemble, timestamp):
    '''
    pointGDF= eodataframe with geometry describing the locations of the different points, nothing else
    rasterDA = xarray.DataArray with 
    '''

    point_latitude, point_longitude = pointGDF.get_coordinates().values
    pointGDF ['rastervalue'] = [x for x in rasterDA.sel(
                                            latitude=point_latitude, 
                                            longitude=point_longitude, 
                                            method='nearest')] 
    return pointGDF

def aggregation (rasterDA, vector, method, nrEnsemble=None, timestamp=None, measure=None): 
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
        Statistical measure to calculate for zonal statistics (e.g., 'mean', 'sum').

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
    vectorGDF = checkVectorFormat(vector, shapeType=method, crs=rasterDA.rio.crs, placement='model')
    if method == 'point': 
        pointgdf = query(rasterDA_single, vectorGDF)
        return pointgdf 
    # elif method == 'bufferpoint': 
    #     return pointgdf
    elif method == 'polygon': 
        admgdf = zonal_stats(rasterDA_single, vectorGDF, measure)
        return admgdf
    else: 
        raise ValueError ("Invalid method, choose 'point', or 'adm_unit'")

