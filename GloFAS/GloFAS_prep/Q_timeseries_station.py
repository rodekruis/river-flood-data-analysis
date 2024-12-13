# creates a timeseries of the discharge (or no timeseries but just static value for discharge threshold), 
# for the location corresponding to a certain station (DNH station)
# do this for the 60 percent probability that it will be exceeded?

import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from GloFAS.GloFAS_prep.aggregation import aggregation
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
from GloFAS.GloFAS_data_extractor.reforecast_dataFetch import get_monthsdays
from GloFAS.GloFAS_data_extractor.forecast_dataFetch import compute_dates_range
from GloFAS.GloFAS_data_extractor.threshold_opener import openThreshold
from GloFAS.GloFAS_data_extractor.unzip_open import openGloFAS, unzipGloFAS
import GloFAS.GloFAS_prep.configuration as cfg

class GloFAS_timeseries: 
    def __init__(self,
                DataDir, 
                leadtime, 
                lakesPath, 
                GloFAS_csv, 
                crs, 
                forecastType='reforecast', 
                start_date=None, 
                end_date=None, 
                IDhead='Station names', 
                percentile=60):
        '''
        start and end date are only necessary when you have forecast, because these loop over specific days in specific years
        whereas reforecast data downloaded is dataarray for each day,month , with inside the dataset the years that you have downloaded it for
        '''
        self.DataDir=DataDir
        self.leadtime = leadtime
        self.crs = crs
        self.lakesPath = lakesPath
        self.forecastType = forecastType
        self.IDhead = IDhead
        self.percentile = percentile
        self.GloFAS_stations_gdf = checkVectorFormat (GloFAS_csv, shapeType='point', crs=self.crs)
        if forecastType == 'reforecast':
            self.MONTHSDAYS = get_monthsdays()
        elif forecastType ==  'forecast':
            self.MONTHSDAYSYEARS = compute_dates_range(start_date, end_date)
        else: 
            raise ValueError (f'no such forecast type integrated in this analysis: {forecastType}')

    def Q_over_time(self):
        '''Concatenate exceedance probability calculations over multiple time steps'''
        
        if self.forecastType == 'reforecast':

            for _, station_row in self.GloFAS_stations_gdf.iterrows():
                stationname = station_row[f'{self.IDhead}']
                station_results = []  # Temporary storage for this station's data
                
                #try:
                for month, day in self.MONTHSDAYS:

                    # Fetch the raster path and open the data
                    Q_da_rasterpath = unzipGloFAS(self.DataDir, self.leadtime, month, day)
                    Q_da = openGloFAS(Q_da_rasterpath, self.leadtime)
                    print(f"Starting for {stationname} in {month}-{day}")
                
                    for timestamp in Q_da.time:
                        
                        startLoop = datetime.now()
                        
                        # Calculate the valid timestamp
                        valid_timestamp = pd.to_datetime(str(timestamp.values)) + timedelta(hours=self.leadtime)
                        timestep_data = {'ValidTime': valid_timestamp}
                        
                        # Aggregate discharge for all ensemble members
                        ensemble_values = []
                        for n, ensemble_member in enumerate(Q_da.number):
                            # Assuming `nrEnsemble` should be 1-based
                            o = n + 1  # For 1-based index
                            
                            # Perform aggregation for this ensemble member
                            aggregated_value = aggregation(
                                Q_da, station_row, 'point',
                                nrEnsemble=o, timestamp=timestamp, IDhead=self.IDhead
                            )

                            # Store the aggregated value in the timestep data
                            timestep_data[f'member_{o}'] = aggregated_value['rastervalue'].iloc[0]
                            ensemble_values.append(float(aggregated_value['rastervalue'].iloc[0]))

                        timestep_data[f'percentile_{self.percentile}'] = np.percentile(ensemble_values, self.percentile)
                        # Append to results
                        station_results.append(timestep_data)
                    
                    print(f"Finished {month}-{day} for station {stationname}")
                
                # except Exception as e:
                #     print(f"Skipping leadtime {self.leadtime} for station {stationname}: {e}")
                #     continue  # Skip the rest of this leadtime and move to the next station
                
                # Save results to CSV after processing all time steps for the station
                station_results_df = pd.DataFrame(station_results)
                station_results_df.to_csv(f'{self.DataDir}/stations/timeseries/discharge_timeseries_{stationname}_{self.leadtime}.csv', index=False)
                print(f"Finished station {stationname}")
            
                


    
if __name__ =='__main__': 
    for leadtime in cfg.leadtimes:
        timeseries = GloFAS_timeseries(DataDir=cfg.DataDir,
                                        leadtime=leadtime, 
                                        lakesPath=cfg.lakesPath, 
                                        GloFAS_csv=cfg.GloFASstations,
                                        crs=cfg.crs,
                                        forecastType='reforecast',
                                        start_date=None,
                                        end_date=None, 
                                        IDhead='Station names',
                                        percentile=(100-(cfg.triggerProb*100))) # percentile is 1 - the probability of exceedence. At 40 percent percentile, tehre is a 60 procent probability of this value being exceeded
        Q_timeseries.Q_over_time()
