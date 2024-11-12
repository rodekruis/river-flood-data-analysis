import os
import numpy as np
import pandas as pd
import configuration as cfg
from aggregation import aggregation
from open_extract import unzipGloFAS, openGloFAS
from vectorCheck import checkVectorFormat # this is correctly imported 
from forecast_dataFetch import compute_dates_range

def aggregate_forecasted(
    START_DATE,
    END_DATE,
    pointvector,
    leadtime=168, # 7 days, default
    DataDir=os.getcwd(),
    IDhead='StationName',
    probability=False,
    probabilityInterval=10,  # in percentile
    output_filename='aggregated.csv',  
    lakesPath=None,
    crs='EPSG:4326'
    ):

    """
    Aggregate forecasted data from GloFAS for the given date range and station IDs.
    
    Arguments:
    - START_DATE (str): The start date for aggregation.
    - END_DATE (str): The end date for aggregation.
    - DataDir (str): Directory to save the output CSV.
    - leadtime (int): Leadtime for the forecast.
    - IDhead (str): Column name for the station ID.
    - probability (bool): Whether to aggregate based on probabilities.
    - probabilityInterval (int): The interval for probabilities (default 10).
    - output_filename (str): Name of the output file (default 'aggregated.csv').

    Returns:
    - None: Writes the aggregated data to a CSV file.
    """

    # Check vector format and retrieve point IDs
    pointvectorMODEL_gdf = checkVectorFormat(pointvector, shapeType='point', crs=crs, placement='real')
    pointIDs = pointvectorMODEL_gdf[IDhead]
    # Generate date range
    dates = compute_dates_range(START_DATE, END_DATE)

    # Initialize DataFrame for aggregation
    if probability:
        probabilities = np.arange(0, 101, probabilityInterval)  # Probability percentiles
        indexheads = [f'{pointID}_{perc}' for pointID in pointIDs for perc in probabilities]
        aggregated_df = pd.DataFrame(np.nan, index=dates, columns=indexheads)
    else:
        aggregated_df = pd.DataFrame(np.nan, index=dates, columns=pointIDs)

    # Process data for each date
    for date in dates:
        # Extract year, month, and day for file path formatting
        year  = date.strftime('%Y')
        month = date.strftime('%m')
        day   = date.strftime('%d')

        # Unzip and open raster data for the current date
        rasterPath = unzipGloFAS(DataDir, leadtime, month, day, year)
        Q_da = openGloFAS(rasterPath, lakesPath, crs)
        
        # Aggregate data for probability case
        if probability:
            totalEnsembleMembers = len(Q_da.number)
            step = int(totalEnsembleMembers / probabilityInterval)  # Ensure integer step
            relevantMembers = np.arange(1, totalEnsembleMembers, step)

            # Aggregate data for each ensemble member at each point
            for nrEnsemble in relevantMembers:
                agg_results = aggregation(Q_da, pointvectorMODEL_gdf, 'point', nrEnsemble=int(nrEnsemble), timestamp=date)
                for perc in probabilities:
                    for pointID in pointIDs:
                        aggregated_df.loc[date, f'{pointID}_{perc}'] = agg_results.get((pointID, int(perc)), np.nan)
        
        # Aggregate data for non-probability case
        else:
            aggregation_Q_gdf= aggregation(Q_da, pointvectorMODEL_gdf, 'point', timestamp=date)
            aggregated_df.loc[date, :] = aggregation_Q_gdf ['rastervalue'].values

    # Write the final aggregated DataFrame to CSV
    aggregated_df.to_csv(f'{DataDir}/{output_filename}')
    print(f'Aggregation complete! Data saved to {DataDir}/{output_filename}')

if __name__=='__main__': 
    aggregate_forecasted('2024-06-26', '2024-11-01', pointvector=cfg.GloFASstations, DataDir=cfg.DataDir)