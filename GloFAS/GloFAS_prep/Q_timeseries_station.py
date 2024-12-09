# creates a timeseries of the discharge (or no timeseries but just static value for discharge threshold), 
# for the location corresponding to a certain station (DNH station)
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from GloFAS.GloFAS_prep.aggregation import aggregation
from GloFAS.GloFAS_data_extractor.reforecast_dataFetch import get_monthsdays
from GloFAS.GloFAS_data_extractor.forecast_dataFetch import compute_dates_range
from GloFAS.GloFAS_data_extractor.threshold_opener import openThreshold
from GloFAS.GloFAS_data_extractor.unzip_open import openGloFAS, unzipGloFAS
import GloFAS.GloFAS_prep.configuration as cfg

def Q_over_time(Q_da, forecastType):
    '''Concatenate exceedance probability calculations over multiple time steps'''
    all_years_data = []
    if forecastType=='reforecast':
        for month, day in self.MONTHSDAYS:
            print(f"Starting for {month}, {day}")
            Q_da_rasterpath = unzipGloFAS (self.DataDir, self.leadtime, month, day)
            Q_da = openGloFAS(Q_da_rasterpath, self.leadtime)  # Open data for the specific month/day
            for timestamp in Q_da.time:  # Iterate over years
                startLoop = datetime.now()
                valid_timestamp = pd.to_datetime(str(timestamp.values)) + timedelta(hours=self.leadtime)
                # Calculate probability
                flooded_df = self.probability(Q_da, self.threshold_gdf, nrCores)
                flooded_df['ValidTime'] = valid_timestamp  # Add timestamp
                all_years_data.append(flooded_df)
                print(f"Time step {valid_timestamp} done, took: {datetime.now() - startLoop}")

    if self.forecastType=='forecast': 
        for date in self.MONTHSDAYSYEARS:
            startLoop = datetime.now()
            year  = date.strftime('%Y')
            month = date.strftime('%m')
            day   = date.strftime('%d')
            Q_da_rasterpath = unzipGloFAS(self.DataDir, self.leadtime, month, day, year)
            Q_da = openGloFAS(Q_da_rasterpath, lakesPath=self.lakesPath, crs=self.crs)
            valid_timestamp = pd.to_datetime(str(timestamp.values)) + timedelta(hours=self.leadtime)
            # Calculate probability
            flooded_df = self.probability(Q_da, self.threshold_gdf, nrCores)
            flooded_df['ValidTime'] = valid_timestamp  # Add timestamp
            all_years_data.append(flooded_df)
            print(f"Time step {valid_timestamp} done, took: {datetime.now() - startLoop}")
    # Concatenate results for all time steps
    floodProb_gdf_concat = pd.concat(all_years_data, ignore_index=True)
    floodProb_gdf_concat = gpd.GeoDataFrame(floodProb_gdf_concat, geometry='geometry')
    
    # Save to file
    # make the save to file so that 
    output_path = f"{self.DataDir}/floodedRP{self.RPyr}yr_leadtime{self.leadtime}.gpkg"
    floodProb_gdf_concat.to_file(output_path, driver="GPKG")
    return floodProb_gdf_concat