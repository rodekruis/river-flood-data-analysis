from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.spatial import cKDTree
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
import GloFAS.GloFAS_prep.configuration as cfg 
from GloFAS.GloFAS_prep.text_formatter import remove_accents
import matplotlib.colors as mcolors
from geopy.distance import geodesic
'''
Matching of different representations of space (points, fields) to each other. 
This can be relevant when considering stations that need to be matched to a administrative unit or vice versa
Or when relating points to points, and then using the closest value to within a radius to find it 
'''
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


def find_corresponding_point_within_box(station_lon, station_lat, ups_area_point_m, glofas_ups_area_xr, stationName, radius_m=10000):
    """
    For a given station point, find the grid cell in the glofas_ups_area_xr
    within a bounding box whose upstream area value is closest to the station's upstream area value.

    Parameters:
        point_x (float): Longitude of the station.
        point_y (float): Latitude of the station.
        ups_area_point_m (float): Upstream area for the specific station (in m^3).
        glofas_ups_area_xr (xr.DataArray): Upstream area values in the model grid (in m^3).
        radius_m (float): Radius around the station to define the bounding box (in meters).

    Returns:
        model_point_x (float): Longitude of the best-matching grid cell.
        model_point_y (float): Latitude of the best-matching grid cell.
        best_area_diff (float): Absolute difference between station and grid cell upstream areas.
    """
    # Approximate degrees per meter (latitude/longitude adjustment)
    print("Latitude range in dataset:", glofas_ups_area_xr.latitude.min().values, glofas_ups_area_xr.latitude.max().values)
    print("Longitude range in dataset:", glofas_ups_area_xr.longitude.min().values, glofas_ups_area_xr.longitude.max().values)

    # simplification of translation of metres to degrees 
    degree_buffer = radius_m / 111000  # 1 degree latitude ~ 111 km
    lat_min = station_lat - degree_buffer
    lat_max = station_lat + degree_buffer
    lon_min = station_lon - degree_buffer
    lon_max = station_lon + degree_buffer

    print(f"Latitude bounds: {lat_min} to {lat_max}")
    print(f"Longitude bounds: {lon_min} to {lon_max}")
    lat_resolution = np.diff(glofas_ups_area_xr.latitude.values).mean()
    lon_resolution = np.diff(glofas_ups_area_xr.longitude.values).mean()

    print(f"Latitude resolution: {lat_resolution} degrees")
    print(f"Longitude resolution: {lon_resolution} degrees")
    # Subset the glofas grid within the bounding box
    subset = glofas_ups_area_xr.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )

    # Extract the coordinates and values from the subset
    lats = subset["latitude"].values
    lons = subset["longitude"].values
    area = subset.values

    area_difference = subset - ups_area_point_m

    lats = subset["latitude"].values
    lons = subset["longitude"].values
    area = subset.values
    # area_difference = abs(area - ups_area_point_m)
    # area_difference.plot() 

        # Extract the coordinates and values from the subset
    lats = subset["latitude"].values
    lons = subset["longitude"].values
    areas = subset.values

    # Create a meshgrid of the coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten arrays for easier iteration
    flat_lons = lon_grid.flatten()
    flat_lats = lat_grid.flatten()
    flat_areas = areas.flatten()

    # Initialize variables to store the best match
    best_area_diff = float('inf')
    best_point = (None, None)

    for lon, lat, area in zip(flat_lons, flat_lats, flat_areas):
            # Compute the absolute difference in upstream area
            area_diff = abs(area - ups_area_point_m)

            # Update the best match if this cell's area is closer to the target value
            if area_diff < best_area_diff:
                best_area_diff = area_diff
                best_point = (lon, lat)

    model_point_lon, model_point_lat = best_point
    ################################################################### plot to visualize, feel free to remove of course
    # Create a new figure
    plt.figure(figsize=(10, 8))
    # Plot the data and retrieve colormap and normalization
    plot = (subset / 1e6).plot(
        cmap="viridis",  # Consistent color scheme
        cbar_kwargs={"label": "Upstream Area (km²)"}  # Add colorbar label  # We'll handle the colorbar separately
    )
    cmap = plt.get_cmap("viridis")  # Same colormap as the plot
    norm = mcolors.Normalize(vmin=subset.min().item(), vmax=subset.max().item())  # Normalize based on subset

    # Get the color for the station point based on upstream area value
    point_color = cmap(norm(ups_area_point_m))  # Map the value to the colormap

    # Overlay the station point
    plt.scatter(
        station_lon,
        station_lat,
        s=200,  # Size of the marker
        edgecolor="black",  # Black border
        facecolor=point_color,  # Fill color based on upstream area value
        linewidth=1.5,  # Border thickness
        label=f"Upstream area documented by DNH at station {stationName} = {ups_area_point_m/1e6:.0f} km²"
        )
    plt.scatter(
        model_point_lon,
        model_point_lat,
        s=200,  # Size of the marker
        edgecolor="white",  # white border
        facecolor='none',  # leave fill in empty
        linewidth=1.5,  # Border thickness
        label=f"GloFAS location corresponding to minimal upstream area difference "
        )
    plt.plot ([],[], ' ', label= f'found within {radius_m/1e3:.0f} km radius (difference={best_area_diff/1e6:.1f} km²)')
    plt.title(f"Subset for Station: {stationName}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.savefig(f'{cfg.DataDir}/GloFAS_station_for_DNH_{stationName}.png')
    plt.show()
    plt.close()

    return model_point_lon, model_point_lat, best_area_diff


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
    ID1: str
        Column name in vector_original identifying the points.
    vector2 : str or GeoDataFrame
        The target vector data (file path or GeoDataFrame) containing points to match with.
    ID2: str
        Column name in vector2 identifying the points.
    vector3 : str or GeoDataFrame, optional
        An additional vector data source for matching (optional).
    ID3: str, optional
        Column name in vector3 identifying the points (optional).
    crs: str, optional
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
        buffer_distance_meters=5000,  
        StationDataDir=Path.cwd(), 
        filename='attributePoints_to_Polygon.csv'
        ):
    # Format input data into GeoDataFrames
    points_gdf = checkVectorFormat(vectorPoint, 'point', crs)
    polygons_gdf = checkVectorFormat(vectorPolygon, 'polygon', crs)

    # Ensure the geometries are aligned with the same CRS
    if points_gdf.crs != polygons_gdf.crs:
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)

    # Calculate buffer distance in degrees
    latitude = polygons_gdf.geometry.centroid.y.mean()
    meters_per_degree = 111320 * np.cos(np.radians(latitude))
    buffer_distance_degrees = buffer_distance_meters / meters_per_degree

    # Buffer polygons and store updated geometries
    expanded_polygons_gdf = polygons_gdf.copy()
    expanded_polygons_gdf['geometry'] = expanded_polygons_gdf.geometry.buffer(buffer_distance_degrees)

    # Initialize a dictionary to track max number of stations per polygon
    max_stations = 0

    # Loop through each polygon to find points within it
    for idx, polygon_row in expanded_polygons_gdf.iterrows():
        # Get the geometry of the current polygon
        polygon_geom = polygon_row.geometry

        # Find points that fall within this polygon
        points_within = points_gdf[points_gdf.geometry.within(polygon_geom)]

        # Collect the IDs of these points
        point_ids = points_within[ID2].tolist()

        # Add dynamic columns for each point
        for i, point_id in enumerate(point_ids, start=1):
            column_name = f'{ID2}_{i}'
            polygons_gdf.at[idx, column_name] = remove_accents(point_id)

        # Track the maximum number of stations for column consistency
        max_stations = max(max_stations, len(point_ids))

    # Fill missing columns with NaN for polygons with fewer stations
    for i in range(1, max_stations + 1):
        column_name = f'{ID2}_{i}'
        if column_name not in polygons_gdf.columns:
            polygons_gdf[column_name] = None

    # Write the updated GeoDataFrame to a CSV file
    output_file = StationDataDir / filename
    polygons_gdf.drop(columns='geometry').to_csv(output_file, index=False)

    return polygons_gdf


if __name__ == '__main__': 
    result = attributePoints_to_Polygon(
        cfg.admPath, 
        cfg.DNHstations, 
        'StationName',  
        crs=cfg.crs,
        buffer_distance_meters=5000, # to tolerate the bolder 
        StationDataDir=cfg.stationsDir,
        filename='DNHstations_in_ADM2.csv'
        )
    
    # matchPoints(cfg.GloFASstations, 'StationName', cfg.googlestations, 'gaugeId', crs=cfg.crs, StationDataDir=cfg.stationsDir)