from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from vectorCheck import checkVectorFormat
import batch_configuration as cfg 

def findclosestpoint(point_x, point_y, target_gdf):
    '''
    Finds the closest point in target_gdf to the given coordinates (point_x, point_y).
    
    Parameters:
    point_x : float
        X-coordinate of the reference point.
    point_y : float
        Y-coordinate of the reference point.
    target_gdf : GeoDataFrame
        A GeoDataFrame containing points to search for the closest one.
        
    Returns:
    Series
        The row in target_gdf corresponding to the closest point.
    '''
    
    # Extract coordinates of all points in target_gdf
    target_coords = np.vstack([target_gdf.geometry.x, target_gdf.geometry.y]).T  # Shape: (n_points, 2)
    tree = cKDTree(target_coords)  # Build KDTree for fast nearest neighbor search
    dist, idx = tree.query([(point_x, point_y)], k=1)  # Find closest point
    return target_gdf.iloc[idx[0]]  # Return row of the closest point

def matchPoints(
            vector_original, 
            ID1, 
            vector2, 
            ID2, 
            vector3=None, 
            ID3=None, 
            crs='EPSG:4326', 
            StationDataDir=Path.cwd()
            ):
    '''
    Matches each point in vector_original with the closest point in vector2, and optionally vector3.
    
    Parameters:
    vector_original : str or GeoDataFrame
        The original vector data (file path or GeoDataFrame) containing the source points.
    ID1 : str
        Column name in vector_original identifying the points.
    vector2 : str or GeoDataFrame
        The target vector data (file path or GeoDataFrame) containing points to match with.
    ID2 : str
        Column name in vector2 identifying the points.
    vector3 : str or GeoDataFrame, optional
        An additional vector data source for matching (optional).
    ID3 : str, optional
        Column name in vector3 identifying the points (optional).
    crs : str, optional
        Coordinate reference system for all data. Defaults to 'EPSG:4326'.
    StationDataDir: str or Path, optional 
        Datadirectory describing where to write the stationdirectory to, default is root
    
    Writes: 
    CSV
        A csv file containing the original point ID and coordinates, and closest point IDs and coordinates from vector2 and optionally vector3.
    
    Returns:
    DataFrame
        A DataFrame containing the original point ID and coordinates, and closest point IDs and coordinates from vector2 and optionally vector3.
    '''
    
    # Format vectors into GeoDataFrames with point geometries and specified CRS
    vector1_gdf = checkVectorFormat(vector_original, 'point', crs)
    vector2_gdf = checkVectorFormat(vector2, 'point', crs)
    vector3_gdf = checkVectorFormat(vector3, 'point', crs) if vector3 is not None else None

    # Initialize an empty list to store results
    match_data = []
    
    # Loop through each point in vector1_gdf and find closest points in vector2_gdf (and optionally vector3_gdf)
    for _, pointrow in vector1_gdf.iterrows():
        # Extract coordinates of the current point
        point_x, point_y = pointrow.geometry.x, pointrow.geometry.y

        # Get the closest point in vector2_gdf
        closest_point_vector2 = findclosestpoint(point_x, point_y, vector2_gdf)

        # Prepare match row with data from vector1 and vector2
        match_row = {
            f'{ID1}': pointrow[ID1],
            f'{ID1}_x': point_x,
            f'{ID1}_y': point_y,
            f'{ID2}': closest_point_vector2[ID2],
            f'{ID2}_x': closest_point_vector2.geometry.x,
            f'{ID2}_y': closest_point_vector2.geometry.y
            }

        # If vector3 is provided, find the closest point in vector3 as well
        if vector3_gdf is not None and ID3 is not None:
            closest_point_vector3 = findclosestpoint(point_x, point_y, vector3_gdf)
            match_row.update({
                f'{ID3}': closest_point_vector3[ID3],
                f'{ID3}_x': closest_point_vector3.geometry.x,
                f'{ID3}_y': closest_point_vector3.geometry.y
            })

        # Append the match row to the list
        match_data.append(match_row)

    # Convert match_data into a DataFrame
    match_df = pd.DataFrame(match_data)
    
    # Write match_df to CSV if needed or return it directly
    match_df.to_csv(f'{StationDataDir}/matchpoints.csv', index=False)
    return match_df


if __name__ == '__main__': 
    matchPoints(cfg.GloFASstations, 'StationName', cfg.googlestations, 'gaugeId', crs=cfg.crs, StationDataDir=cfg.stationsDir)