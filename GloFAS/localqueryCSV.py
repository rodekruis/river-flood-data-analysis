
# I apologize for these imports, but it had to be this way 
import GloFAS.GloFAS_prep.configuration as cfg
from GloFAS.GloFAS_prep.aggregation import aggregation
from GloFAS.GloFAS_data_extractor.unzip_open import unzipGloFAS, openGloFAS
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat # this is correctly imported 
from GloFAS.GloFAS_data_extractor.forecast_dataFetch import compute_dates_range
import os
import numpy as np
import pandas as pd

def aggregate_forecasted(
    START_DATE,
    END_DATE,
    vector,
    leadtime=168, # 7 days, default
    DataDir=os.getcwd(),
    IDhead='StationName',
    method='point',
    probability=False,
    probabilityInterval=10,  # in percentile
    output_filename='aggregated.csv',  
    lakesPath=None,
    crs='EPSG:4326',
    measure='max'
    ):

    """
    Aggregate forecasted data from GloFAS for the given date range and station IDs / polygons
    
    Arguments:
    - START_DATE (str): The start date for aggregation.
    - END_DATE (str): The end date for aggregation.
    - vector (str/Path/gpkg): describing the units on which to aggregate
    - DataDir (str): Directory to save the output CSV.
    - leadtime (int): Leadtime for the forecast.
    - IDhead (str): Column name for the station ID / administrative unit name
    - method (str): 'point' or 'polygon'
    - probability (bool): Whether to aggregate for different ensemble members (uncertainties).
    - probabilityInterval (int): The interval for probabilities (default 10).
    - output_filename (str): Name of the output file (default 'aggregated.csv').
    - measure (str): in case of 'polygon' method: what measure to use?
    Returns:
    - None: Writes the aggregated data to a CSV file.
    Csv: is of long format with 
        rows for each  ISSUE date (so not date of validity), 
        columns: each point/administrative unit
        and then filled with the values that describe the raster at that point, or the max/mean/min in the polygon
    """

    # Check vector format and retrieve point IDs
    vector_gdf = checkVectorFormat(vector, shapeType=method, crs=crs, placement='model')
    IDs = vector_gdf[IDhead]
    # Generate date range
    dates = compute_dates_range(START_DATE, END_DATE)

    # Initialize DataFrame for aggregation
    if probability:
        probabilities = np.arange(0, 101, probabilityInterval)  # Probability percentiles
        indexheads = [f'{ID}_{perc}' for ID in IDs for perc in probabilities]
        aggregated_df = pd.DataFrame(np.nan, index=dates, columns=indexheads)
    else:
        aggregated_df = pd.DataFrame(np.nan, index=dates, columns=IDs)

    # Process data for each date
    for date in dates:
        # Extract year, month, and day for file path formatting
        year  = date.strftime('%Y')
        month = date.strftime('%m')
        day   = date.strftime('%d')

        # Unzip and open raster data for the current date
        rasterPath = unzipGloFAS(DataDir, leadtime, month, day, year)
        Q_da = openGloFAS(rasterPath, lakesPath, crs, forecastType='forecasted')
        
        # Aggregate data for probability case
        if probability:
            totalEnsembleMembers = len(Q_da.number)
            step = int(totalEnsembleMembers / probabilityInterval)  # Ensure integer step
            relevantMembers = np.arange(1, totalEnsembleMembers, step)

            # Aggregate data for each ensemble member at each point
            for nrEnsemble in relevantMembers:
                agg_results = aggregation(Q_da, vector_gdf, method , nrEnsemble=int(nrEnsemble), timestamp=date, measure=measure, IDhead=IDhead)
                for perc in probabilities:
                    for pointID in pointIDs:
                        aggregated_df.loc[date, f'{pointID}_{perc}'] = agg_results.get((pointID, int(perc)), np.nan)
        
        # Aggregate data for non-probability case
        else:
            aggregation_Q_gdf= aggregation(Q_da, vector_gdf, method, timestamp=date, measure=measure, IDhead=IDhead)
            if method=='point':
                aggregated_df.loc[date, :] = aggregation_Q_gdf ['rastervalue'].values
            if method=='polygon':
                aggregated_df.loc[date, :] = aggregation_Q_gdf [f'{measure}'].values

    # Write the final aggregated DataFrame to CSV
    aggregated_df.to_csv(f'{DataDir}/{output_filename}')
    print(f'Aggregation complete! Data saved to {DataDir}/{output_filename}')

if __name__=='__main__': 
    aggregate_forecasted(
                        '2024-06-26', 
                        '2024-11-01', 
                        vector=cfg.admPath, 
                        DataDir=cfg.DataDir, 
                        leadtime=168, 
                        IDhead=f'ADM{cfg.adminLevel}_FR', 
                        method='polygon', 
                        output_filename='aggregated_admin.csv', 
                        lakesPath=cfg.lakesPath)