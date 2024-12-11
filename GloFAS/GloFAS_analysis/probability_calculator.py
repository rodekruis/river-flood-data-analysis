import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from GloFAS.GloFAS_prep.aggregation import aggregation
from GloFAS.GloFAS_data_extractor.reforecast_dataFetch import get_monthsdays
from GloFAS.GloFAS_data_extractor.forecast_dataFetch import compute_dates_range
from GloFAS.GloFAS_data_extractor.threshold_opener import openThreshold
from GloFAS.GloFAS_data_extractor.unzip_open import openGloFAS, unzipGloFAS
import GloFAS.GloFAS_prep.configuration as cfg

class FloodProbabilityProcessor:
    def __init__(self, 
                DataDir=cfg.DataDir, 
                leadtime=168, 
                area=cfg.MaliArea, 
                lakesPath=cfg.lakesPath, 
                crs=cfg.crs, 
                adminLevel=None, 
                forecastType='reforecast', 
                measure='max', 
                start_date=None, 
                end_date=None, 
                nrCores=4, 
                comparisonShape='polygon'):
                
        # start and end_date are only necessary if your forecastType is not 'reforecast' but 'forecast'
        
        self.leadtime = leadtime
        self.DataDir = DataDir
        self.crs = crs
        self.RPyr =RPyr
        self.area =area
        self.lakesPath = cfg.lakesPath
        self.forecastType = forecastType
        
        if forecastType == 'reforecast':
            self.MONTHSDAYS = get_monthsdays()
        elif forecastType ==  'forecast':
            self.MONTHSDAYSYEARS = compute_dates_range(start_date, end_date)
        # Initializations that rely on data
        self.reference_rasterPath = unzipGloFAS(self.DataDir, self.leadtime, '01', '07')
        self.reference_Q_da = openGloFAS(self.reference_rasterPath, self.lakesPath, self.crs)
        self.threshold_da = openThreshold(self.DataDir, self.crs, self.RPyr, self.area, self.reference_Q_da)

        self.comparisonShape = comparisonShape
        if comparisonShape == 'polygon':
            self.measure = measure
            self.threshold_gdf = aggregation(self.threshold_da, adminPath, 'polygon', measure=self.measure)
            self.adminLevel = adminLevel

        elif comparisonShape == 'point': 
            self.threshold_gdf = aggregation(self.threshold_da, GloFAS_stations)
        self.nrCores = nrCores
        
    def exceedance(self, Q_da, threshold_gdf, nrEnsemble):
        '''Check exceedance of threshold values for a single ensemble member
        threshold_gdf should be in same shape as eventual outcme, so if comparisonShape=polygon, threshold_gdf should be in polygon'''
        if self.comparisonShape=='polygon':
            GloFAS_gdf = aggregation(Q_da, threshold_gdf, 'polygon', nrEnsemble, measure=self.measure)
            exceedance = GloFAS_gdf[f'{self.measure}'] > threshold_gdf[f'{self.measure}']
        elif self.comparisonShape=='point': # assuming this then refers to aggregation to a point
            GloFAS_gdf = aggregation (Q_da, threshold_gdf, 'point', nrEnsemble)
            exceedance = GloFAS_gdf ['rastervalue'] > threshold_gdf['rastervalue']
        else: 
            raise ValueError (f"Not a valid shape : {self.comparisonShape}, pick between 'polygon' and 'point'")
        return exceedance

    def probability(self, Q_da, threshold_gdf, nrCores):
        '''Calculate exceedance probability across ensemble members for a single time step'''
        nrMembers = len(Q_da.number)
        exceedance_count = np.zeros(len(threshold_gdf), dtype=int)
        
        # Calculate exceedances for each ensemble in parallel
        results = Parallel(n_jobs=nrCores)(delayed(self.exceedance)(Q_da, threshold_gdf, ensemble) for ensemble in Q_da.number)
        for result in results:
            exceedance_count += result
        probability = exceedance_count / nrMembers

        # Prepare results as GeoDataFrame
        flooded_df = threshold_gdf.copy()
        flooded_df['ExceedanceCount'] = exceedance_count
        flooded_df['Probability'] = probability
        return flooded_df

    def concatenate_prob(self, Q_da, nrCores):
        '''Concatenate exceedance probability calculations over multiple time steps'''
        all_years_data = []
        if self.forecastType=='reforecast':
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
