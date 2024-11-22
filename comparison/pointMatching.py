from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
import GloFAS.GloFAS_prep.configuration as cfg 
from GloFAS.GloFAS_prep.text_formatter import remove_accents

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

def attributePoints_to_Polygon(
        vectorPolygon, 
        vectorPoint, 
        ID2, 
        crs='EPSG:4326', 
        border_tolerance=5000,  # Buffer distance in meters
        StationDataDir=Path.cwd(), 
        filename='attributePoints_to_Polygon.csv'
        ):
    '''
    Assigns points from vector_original to polygons in vector2, adding a new column in vector2 that contains the IDs of all points
    from vector_original that fall within each polygon or within a specified buffer distance.

    Parameters:
    vectorPolygon : str or GeoDataFrame
        The original vector data (file path or GeoDataFrame) containing the source polygon.
    ID1 : str
        Column name in vector_original identifying the points.
    vectorPoint : str or GeoDataFrame
        The target vector data (file path or GeoDataFrame) containing points.
    ID2 : str
        Column name in vector2 identifying the polygons.
    crs : str, optional
        Coordinate reference system for all data. Defaults to 'EPSG:4326'.
    border_tolerance : float, optional
        Distance in meters to expand the polygons for including nearby points. Defaults to 5000 (5 km).
    StationDataDir : str or Path, optional
        Directory where the output CSV file will be saved. Default is the current working directory.

    Writes:
    CSV
        A CSV file containing the polygon ID and the IDs of points within or near each polygon.

    Returns:
    GeoDataFrame
        A GeoDataFrame containing the polygon data with an additional column for point IDs.
    '''
    
    # Format input data into GeoDataFrames
    points_gdf = checkVectorFormat(vectorPoint, 'point', crs)
    polygons_gdf = checkVectorFormat(vectorPolygon, 'polygon', crs)

    # Ensure the geometries are aligned with the same CRS
    if points_gdf.crs != polygons_gdf.crs:
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)

    # Apply a buffer to the polygons
    expanded_polygons_gdf = polygons_gdf.copy()
    expanded_polygons_gdf['geometry'] = expanded_polygons_gdf.geometry.buffer(border_tolerance)

    # Initialize a new column in the polygons GeoDataFrame to store point IDs
    polygons_gdf[f'{ID2}'] = None

    # Loop through each polygon to find points within it
    for idx, polygon_row in expanded_polygons_gdf.iterrows():
        # Get the geometry of the current buffered polygon
        expanded_polygon_geom = polygon_row.geometry

        # Find points that fall within this expanded polygon
        points_within = points_gdf[points_gdf.geometry.within(expanded_polygon_geom)]

        # Collect the IDs of these points
        point_ids = points_within[ID2].tolist()
        
        # Update the original polygons GeoDataFrame with the list of point IDs
        nrstations_inpolygon = len(point_ids)
        if nrstations_inpolygon > 0:
            polygons_gdf.at[idx, f'{ID2}'] = remove_accents(point_ids[0])
        if nrstations_inpolygon > 1: 
            for stationnr in range(nrstations_inpolygon):
                polygons_gdf.at[idx, f'{ID2}_{stationnr}'] = remove_accents(point_ids[stationnr-1])

    # Write the updated GeoDataFrame to a CSV file
    output_file = StationDataDir / f'{filename}'
    polygons_gdf.drop(columns='geometry').to_csv(output_file, index=False)
    return polygons_gdf


if __name__ == '__main__': 
    result = attributePoints_to_Polygon(
    cfg.admPath, 
    cfg.DNHstations, 
    'StationName', 
    crs=cfg.crs, 
    border_tolerance=5000, # to tolerate the bolder 
    StationDataDir=cfg.stationsDir,
    filename='DNHstations_in_ADM2.csv'
    )
    
    # matchPoints(cfg.GloFASstations, 'StationName', cfg.googlestations, 'gaugeId', crs=cfg.crs, StationDataDir=cfg.stationsDir)