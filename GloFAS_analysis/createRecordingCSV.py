from aggregation import aggregation 
from open_extract import unzipGloFAS, openGloFAS
from vectorCheck import checkVectorFormat
from forecast_dataFetch import compute_dates_range
import numpy as np
import pandas as pd 
import batch_configuration as cfg 

# Define constants
START_DATE = '2024-06-26'
END_DATE = '2024-11-01'
leadtime = 168
IDhead = 'StationName'

# Check vector format and retrieve point IDs
pointvectorMODEL_gdf = checkVectorFormat(cfg.GloFASstations, shapeType='point', crs=cfg.crs, placement='model')
pointIDs = pointvectorMODEL_gdf[IDhead]

# Define probability percentiles
probabilities = np.arange(0, 101, 10)  # From 0% to 100% in steps of 10

# Define column headers for output dataframes
indexheads = []
for pointID in pointIDs: 
    for perc in probabilities: 
        indexheads.append(f'{pointID}_{perc}')  # Combining station ID and probability for unique column names

# Generate date range
dates = compute_dates_range(START_DATE, END_DATE)

# Initialize DataFrames
# Use np.nan to fill initial values, and set up DataFrames with proper indexing and column names
aggregated_df = pd.DataFrame(np.nan, index=dates, columns=pointIDs)
ensemblemembers_aggregated_df = pd.DataFrame(np.nan, index=dates, columns=indexheads)

# Process data for each date
for date in dates:
    # Extract year, month, and day for file path formatting
    year  = date.strftime('%Y')
    month = date.strftime('%m')
    day   = date.strftime('%d')

    # Unzip and open raster data for the current date
    rasterPath = unzipGloFAS(cfg.DataDir, leadtime, month, day, year)
    Q_da = openGloFAS(rasterPath, cfg.lakesPath)
    
    # Determine ensemble members based on the number of probabilities
    totalEnsembleMembers = len(Q_da.number)
    step = int(totalEnsembleMembers / 10)  # Ensure step is integer
    relevantMembers = np.arange(1, totalEnsembleMembers, step)

    # Aggregate data for each ensemble member at each point
    for nrEnsemble in relevantMembers:
        # Perform the aggregation and assign to the respective date and column in `ensemblemembers_aggregated_df`
        agg_results = aggregation(Q_da, pointvectorMODEL_gdf, 'point', nrEnsemble=int(nrEnsemble), timestamp=date)
        for idx, (pointID, perc) in enumerate(zip(pointIDs, probabilities)):
            ensemblemembers_aggregated_df.loc[date, f'{pointID}_{perc}'] = agg_results.get((pointID, int(perc)), np.nan)

# Write the final aggregated DataFrame to CSV
ensemblemembers_aggregated_df.to_csv(f'{cfg.DataDir}/aggregated.csv')
