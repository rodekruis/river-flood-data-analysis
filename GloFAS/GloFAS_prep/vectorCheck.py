import xarray as xr 
import geopandas as gpd
import pandas as pd
from pathlib import Path
from rasterstats import zonal_stats
from shapely import wkt
import rioxarray as rio

def df_from_txt_or_csv (vector): 
    if vector.endswith ('.csv'):
        df = pd.read_csv(vector)
    elif vector.endswith ('.txt'):
        df = pd.read_csv (vector, sep="; ", header=0)
    return df

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
        elif vector.endswith(('.csv', '.txt')):
            # Load CSV and convert to GeoDataFrame based on shapeType
            df = df_from_txt_or_csv (vector)
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
                    elif {'Lon', 'Lat'}.issubset(df.columns):
                        vectorGDF = gpd.GeoDataFrame(
                            df,
                            geometry=gpd.points_from_xy(df['Lon'], df['Lat']),
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


