import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import unidecode
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from GloFAS.GloFAS_prep.text_formatter import capitalize, remove_accents
from PTM.PTM_prep.ptm_events import ptm_events
from comparison.observation.HydroImpact import events_per_adm#, loop_over_stations
from GloFAS.GloFAS_analysis.flood_definer import FloodDefiner
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
import GloFAS.GloFAS_prep.config_comp as cfg
'''
To calculate the performance, run: from the PredictedToImpactPerformanceAnalyzer 
            analyzer.matchImpact_and_Trigger()
            analyzer.calculateCommunePerformance() 
then, calculate 
'''
class PredictedToImpactPerformanceAnalyzer:
    def __init__(self, 
                DataDir, 
                RPyr, 
                impactData, 
                triggerProb, 
                startYear, 
                endYear, 
                years, 
                PredictedEvents_gdf, 
                comparisonType, 
                actionLifetime, 
                model, 
                leadtime,
                adminLevel=None, # depends on wheter aggregating on stationcoordinates or adminpath
                adminPath=None,
                stationcoordinates=None
                ):
        """
        Initialize the FloodPerformanceAnalyzer class with the required data.
        
        Parameters:
            floodCommune_gdf (GeoDataFrame): GeoDataFrame containing flood predictions.
            impact_gdf (GeoDataFrame): GeoDataFrame containing actual flood impacts. 
                                        should make this a csv possiblity
            triggerProb (float): Probability threshold to classify flood triggers.
        """

        self.triggerProb = triggerProb
        self.adminLevel = adminLevel
        self.DataDir = DataDir 
        self.RPyr = RPyr 
        self.startYear = startYear
        self.endYear = endYear
        self.years = years
        self.actionLifetime = actionLifetime

        self.impactData = impactData
        self.model = model
        self.leadtime = leadtime

        self.comparisonType = comparisonType # Observation vs Impact
        self.comparisonDir = Path(f'{self.DataDir}/{self.model}/{comparisonType}')
        self.comparisonDir.mkdir (parents=True, exist_ok=True)
        if comparisonType == 'Impact':
            gdf_shape = gpd.read_file(adminPath)
            self.spatial_unit = f'ADM{self.adminLevel}'
            self.gdf_shape = gdf_shape[gdf_shape[f'{self.spatial_unit}_FR'].apply(lambda x: isinstance(x, str))]
            #rename shapefile and make capital
            self.gdf_shape.rename(columns={f'{self.spatial_unit}_FR':f'{self.spatial_unit}'}, inplace=True)
            self.gdf_shape[f'{self.spatial_unit}'] = self.gdf_shape[f'{self.spatial_unit}'].apply(lambda x: unidecode.unidecode(x).upper())
        elif comparisonType == 'Observation':
            self.spatial_unit = f'StationName'
            self.gdf_shape = checkVectorFormat (stationcoordinates, shapeType='point')
            self.gdf_shape[self.spatial_unit]=self.gdf_shape[self.spatial_unit].apply(remove_accents)
            self.gdf_shape[self.spatial_unit]= self.gdf_shape[self.spatial_unit].apply(capitalize)

        else:
            print (f'no such comparisontype: {comparisonType}')

        if self.model == 'PTM':
            self.PredictedEvents = self.openPTM_events(PredictedEvents_gdf)
            
        else:
            self.PredictedEvents = self.openModel_events(PredictedEvents_gdf)
            


        self.impact_gdf = self.openObservedImpact_gdf(comparisonType)

         # days before and after validtime the prediction is also valid

    

    def openObservedImpact_gdf(self, typeImpact):
        # Load the data
        if isinstance(self.impactData, str): 
            if self.impactData.endswith('.csv'):
                delimiter = ";" if typeImpact == "Impact" else ","
                df = pd.read_csv(self.impactData, delimiter=delimiter)
            else:
                df = gpd.read_file(self.impactData)
        else:  # assuming other option is a DataFrame
            df = self.impactData
        
        # Add the comparison column
        df[f'{comparisonType}'] = 1

        # Handle dates
        if 'Start Date' in df and 'End Date' in df:  # Ensure these columns exist
            df['Start Date'] = df['Start Date'].astype(str).str.split(' ', expand=True)[0]
            df['End Date'] = df['End Date'].astype(str).str.split(' ', expand=True)[0]
            df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
            df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
        else: 
            raise ValueError (f"Columns 'End Date' and 'Start Date' are not in columns for observed impacts: {df.columns}")
        # Filter rows based on year range
        df_filtered = df[(df['End Date'].dt.year >= self.startYear) & (df['End Date'].dt.year < self.endYear)]
        # Remove non-string entries from ADM columns
        df_filtered = df_filtered[df_filtered[self.spatial_unit].apply(lambda x: isinstance(x, str))]
        df_filtered[self.spatial_unit] = df_filtered[self.spatial_unit].apply(lambda x: unidecode.unidecode(x).upper())
        # Merge the CSV data with the shapefile data
        impact_gdf = pd.merge(self.gdf_shape, df_filtered, how='left', left_on=f'{self.spatial_unit}', right_on=f'{self.spatial_unit}')
        
        return impact_gdf

    def openModel_events (self, predictedEvents_gdf): 
                # Load the data

        try:
            if isinstance(predictedEvents_gdf, (str)):
                if predictedEvents_gdf.endswith('.csv'):
                    df = pd.read_csv(predictedEvents_gdf)  # delimiter set back to ',' if csv is not generated by Tijn
                else: 
                    df = gpd.read_file(predictedEvents_gdf)
                    df = df.drop(columns='geometry')
            elif isinstance(predictedEvents_gdf, pd.DataFrame):
                df = predictedEvents_gdf
            else:
                raise ValueError("Invalid type for predictedEvents_gdf")
        except Exception as e: 
            print(f'No model prediction data for leadtime: {self.leadtime}, RP: {self.RPyr}. Error: {e}')
            df = pd.DataFrame()  # df is assigned and returned empty here
            return df
        
        df[f'Event'] = 1
        if 'Start Date' in df and 'End Date' in df:
            df['End Date'] = pd.to_datetime(df['End Date']) + timedelta(days=self.actionLifetime) #, format='%d/%m/%Y', errors='coerce')
            df['Start Date'] = pd.to_datetime(df['Start Date'])#, format='%d/%m/%Y', errors= 'coerce')
        else: 
            raise ValueError(f"Columns 'End Date' and 'Start Date' are not in columns for predictive events: {df.columns}")

        # Filter rows between 2004 and 2022 ()
        df_filtered = df[(df['End Date'].dt.year >= self.startYear) & (df['End Date'].dt.year < self.endYear)]

        # Remove non-string entries from ADM columns
        df_filtered = df_filtered[df_filtered[self.spatial_unit].apply(lambda x: isinstance(x, str))]
        if df_filtered.empty: 
            print (f'no flood events predicted for return period: {self.RPyr}')
            return df_filtered
        else:
            df_filtered[self.spatial_unit] = df_filtered[self.spatial_unit].apply(lambda x: unidecode.unidecode(x).upper().strip().replace(" ", ""))

            # Merge the CSV data with the shapefile data
            print ('df filtered')
            print(df_filtered[self.spatial_unit].unique())
            print ('gdf shape')
            print(self.gdf_shape[self.spatial_unit].unique())
            modelevents= pd.merge(self.gdf_shape, df_filtered, how='left', left_on=self.spatial_unit, right_on=self.spatial_unit)
        return modelevents

    def openPTM_events (self, predictedEvents_gdf): 
        # Load the data
        if isinstance (predictedEvents_gdf, str):
            if predictedEvents_gdf.endswith('.csv'):
                df = pd.read_csv(predictedEvents_gdf) # delimiter set back to ','if csv is not generated by Tijn 
            else: 
                df = gpd.read_file(predictedEvents_gdf)
        if isinstance (predictedEvents_gdf, pd.DataFrame):
            df = predictedEvents_gdf

        # only select the events within the right start and end date, as well as for the right leadtime
        df[f'Event']=1
        df['End Date'] = pd.to_datetime(df['End Date']) + timedelta(days=self.actionLifetime) #, format='%d/%m/%Y', errors='coerce')
        df['Start Date'] = pd.to_datetime(df['Start Date'])#, format='%d/%m/%Y', errors= 'coerce')
        
        # Filter rows between 2004 and 2022, as well as for the leadtime of interest
        df_filtered = df[(df['End Date'].dt.year >= self.startYear) & (df['End Date'].dt.year < self.endYear) & (df['LeadTime'] == self.leadtime)]
        # Remove non-string entries from ADM columns
        df_filtered = df_filtered[df_filtered[self.spatial_unit].apply(lambda x: isinstance(x, str))]

        
        if df_filtered.empty: 
            print (f'no flood events predicted for return period: {self.RPyr}, leadtime: {self.leadtime}')
            return df_filtered # which will then be empty and just lead to no calculation of performance
        else:
            df_filtered[self.spatial_unit] = df_filtered[self.spatial_unit].apply(lambda x: unidecode.unidecode(x).upper())
                    # Merge the CSV data with the shapefile data
            modelevents= pd.merge(self.gdf_shape, df_filtered, how='left', left_on=self.spatial_unit, right_on=self.spatial_unit)
            return modelevents

    def _check_ifmatched (self, commune, startdate, enddate):
        match = self.impact_gdf[
                        (self.impact_gdf[self.spatial_unit] == commune) & 
                        (self.impact_gdf['Start Date'] <= enddate ) &
                        (self.impact_gdf['Start Date'] >= startdate )
                        ]
        return 1 if not match.empty else 0
    def clean_and_add_modelprediction (self, PredictedEvents_gdf): 
        """
        ONLY CLEAN ONCE MATCHES between impact-predicted HAVE BEEN MADE
        Clean the mdoel's dataset by:
        - Removing entries where 'Event' is 0
        - Removing entries where there is no correspondence to the impact data (as to prevent double entries where matches have been made),
        - Removing entries for communes with no recorded impact in that year.
        After cleaning, add the remaining rows from GloFASCommune_gdf to impact_gdf,
        mapping 'Start Date' to 'Start Date', 'End Date' to 'End Date', and
        self.spatial_unit to self.spatial_unit.
        """

        PredictedEvents_gdf['Remove'] = PredictedEvents_gdf.apply(
            lambda row: self._check_ifmatched(
                row[self.spatial_unit], 
                row['Start Date'], 
                row['End Date']),
                axis=1
            )

         # Keep entries from the original GloFASCommune_gdf where there was not yet a match
        PredictedEvents_gdf = PredictedEvents_gdf[PredictedEvents_gdf['Remove'] != 1] # remove was a 1, so we must keep everything that is not 1
        
        #NO YEAR IMPACT? --> no checking : this is the piece of code: DONT DO THIS FOR OBSERVATIONAL DATA 
        if self.comparisonType == 'Impact':
            for year in self.years:
                start_dry_season = pd.to_datetime(f'{year}-03-01')
                end_dry_season = pd.to_datetime(f'{year}-05-31')

                # Remove rows that fall within the dry season
                # impact data
                self.impact_gdf = self.impact_gdf[~(
                    (self.impact_gdf['Start Date'] >= start_dry_season) & 
                    (self.impact_gdf['End Date'] <= end_dry_season)
                )]
                # nonmatched model prediction events
                PredictedEvents_gdf = PredictedEvents_gdf[~(
                    (PredictedEvents_gdf['Start Date'] >= start_dry_season) & 
                    (PredictedEvents_gdf['End Date'] <= end_dry_season)
                )]

                # general removal of missing impact data 
                start_of_year = pd.to_datetime(f'{year}-01-01')
                end_of_year = pd.to_datetime(f'{year}-12-31')

                # Find communes that recorded an impact in the given year
                recorded_impacts = self.impact_gdf[
                    (self.impact_gdf['Start Date'] >= start_of_year) & 
                    (self.impact_gdf['End Date'] <= end_of_year)
                ][self.spatial_unit].unique()

                # Keep only rows in PredictedEvents_gdf where an impact was recorded for that commune
                PredictedEvents_gdf = PredictedEvents_gdf[~(
                    (PredictedEvents_gdf['Start Date'] >= start_of_year) & 
                    (PredictedEvents_gdf['End Date'] <= end_of_year) & 
                    (~PredictedEvents_gdf[self.spatial_unit].isin(recorded_impacts))
                )]
                

                
        # Prepare the remaining rows to be added to impact_gdf
        remaining_rows = PredictedEvents_gdf.copy()
        
        # Rename columns to match impact_gdf structure
        remaining_rows = remaining_rows.rename(
            columns={
                'Start Date': 'Start Date',
                'End Date': 'End Date',
                self.spatial_unit: self.spatial_unit,
                'StationName': 'StationName',
                'LeadTime':'LeadTime'})
                
        remaining_rows [f'{comparisonType}'] = 0 # we have established before that these are not matching to any impact data, so the impact at these remaining rows are 0 
        # Append the renamed rows to impact_gdf
        self.impact_gdf = pd.concat([self.impact_gdf, remaining_rows], ignore_index=True)

    def _check_impact(self, PredictedEvents_gdf, commune, startdate, enddate):
        '''Check if impact that has happened in the commune between given dates is RECORDED by glofas.'''
        match = PredictedEvents_gdf[
                                (PredictedEvents_gdf[self.spatial_unit] == commune) & 
                                (PredictedEvents_gdf['Start Date'] <= enddate ) &
                                (PredictedEvents_gdf['End Date'] >= startdate) &
                                (PredictedEvents_gdf['Event']==1)
                                ]
        return 1 if not match.empty else 0

    
    def matchImpact_and_Trigger(self):
        """
        Add 'Impact' or observation and 'Event' columns to the floodCommune_gdf,
        and calculate matches only when relevant data exists.
        """
        if self.PredictedEvents.empty or self.impact_gdf.empty:
            print("Skipped matching impacts and triggers as one of the datasets is empty.")
            return

        # Add Impact column using the check impact date (which only works on the impact_gdf)
        self.impact_gdf['Event'] = self.impact_gdf.apply(
            lambda row: self._check_impact(self.PredictedEvents, row[self.spatial_unit], row['Start Date'], row['End Date']),
            axis=1
        )
        # Clean and add GloFAS to self.impact_gdf
        self.clean_and_add_modelprediction(self.PredictedEvents)
        # Save results

        # self.impact_gdf.drop_duplicates (inplace=True)
        geometry_columns = [col for col in self.impact_gdf.columns if self.impact_gdf[col].dtype.name == 'geometry']

        # Drop all but one geometry column
        if len(geometry_columns) > 1:
            self.impact_gdf = self.impact_gdf.drop(columns=geometry_columns[1:])  # Keep only the first geometry column

        # Ensure the remaining column is set as active geometry
        self.impact_gdf.set_geometry(geometry_columns[0], inplace=True)

        self.impact_gdf.to_file(f"{self.DataDir}/{self.comparisonType}/model_vs{self.comparisonType}RP{self.RPyr:.1f}_yr.gpkg")


    def calc_performance_scores(self, obs, pred):
        '''Calculate performance scores based on observed and predicted values,
        including accuracy and precision'''
        print (f'obs: {sum(obs)}, pred: {sum(pred)}')
        # Initialize counters
        hits = 0           # True Positives
        misses = 0         # False Negatives
        false_alarms = 0   # False Positives

        # Calculate hits misses fa 
        for truth, prediction in zip(obs, pred):
            if truth == 1 and prediction == 1:  # True Positive
                hits += 1
            elif truth == 1 and prediction == 0:  # False Negative
                misses += 1
            elif truth == 0 and prediction == 1:  # False Positive
                false_alarms += 1 # there is no use in calculating correct negatives as there are many
        
        # Calculate metrics
        sum_predictions = false_alarms + hits
        if sum_predictions == 0: 
            output = {
                'pod':  np.nan,  # Probability of Detection
                'far': np.nan,  # False Alarm Ratio
                #'pofd': false_alarms / (false_alarms + correct_negatives) if false_alarms + correct_negatives != 0 else np.nan,  # Probability of False Detection
                'csi': np.nan,  # Critical Success Index
                # 'accuracy': (hits + correct_negatives) / (hits + correct_negatives + false_alarms + misses) if hits + correct_negatives + false_alarms + misses != 0 else np.nan,  # Accuracy
                'precision': np.nan,
                'TP': hits,  
                'FN': misses,
                'FP': false_alarms
            }
        else: 
            print (f'hits: {hits}, misses: {misses}, false alarms: {false_alarms}, total {comparisonType} = {sum(obs==1)}, should be equal to {misses}+{hits}={misses+hits}')
            output = {
                'pod': hits / (hits + misses) if hits + misses != 0 else np.nan,  # Probability of Detection
                'far': false_alarms / (hits + false_alarms) if hits + false_alarms != 0 else np.nan,  # False Alarm Ratio
                #'pofd': false_alarms / (false_alarms + correct_negatives) if false_alarms + correct_negatives != 0 else np.nan,  # Probability of False Detection
                'csi': hits / (hits + false_alarms + misses) if hits + false_alarms + misses != 0 else np.nan,  # Critical Success Index
                # 'accuracy': (hits + correct_negatives) / (hits + correct_negatives + false_alarms + misses) if hits + correct_negatives + false_alarms + misses != 0 else np.nan,  # Accuracy
                'precision': hits / (hits + false_alarms) if hits + false_alarms != 0 else np.nan,
                'TP': hits,  
                'FN': misses,
                'FP': false_alarms
            }

        return pd.Series(output)

    def calculateCommunePerformance(self):
        """
        Calculate the performance scores for each commune and merge them back into the GeoDataFrame,
        only if data is available.
        """
        if self.PredictedEvents.empty or self.impact_gdf.empty:
            print("Skipped performance calculation as no impact data is available.")
            return

        # Group by 'Commune' and calculate performance scores for each group
        scores_by_commune = self.impact_gdf.groupby(self.spatial_unit).apply(
            lambda x: self.calc_performance_scores(x[f'{self.comparisonType}'], x['Event'])
        )
        scores_byCommune_gdf = self.gdf_shape.merge(scores_by_commune, on=f'{self.spatial_unit}')
        scores_byCommune_gdf.to_file(f"{self.DataDir}/{self.model}/{self.comparisonType}/scores_byCommuneRP{self.RPyr:.1f}_yr_leadtime{self.leadtime}.gpkg")
        scores_byCommune_gdf.drop(columns='geometry').to_csv(f"{self.DataDir}/{self.model}/{self.comparisonType}/scores_byCommuneRP{self.RPyr:.1f}_yr_leadtime{self.leadtime}.csv")
        return scores_byCommune_gdf

if __name__=='__main__':
    # impact_csv = f'{cfg.DataDir}/Impact_data/impact_events_per_admin_529.csv'
    # comparisonType ='Impact'
    # for RPyr in cfg.RPsyr: 
    #     # PTM_events = f'{cfg.DataDir}/PTM/floodevents_admUnit_RP{RPyr}yr.csv'
    
    #     PTM_events_per_adm = events_per_adm(cfg.DataDir, cfg.admPath, cfg.adminLevel, cfg.DNHstations, cfg.stationsDir, ptm_events_df, 'PTM', RPyr)
    #     for leadtime in cfg.leadtimes:
    #         # print (readcsv(f"{DataDir}/Données partagées - DNH Mali - 20s19/Donne╠ües partage╠ües - DNH Mali - 2019/De╠übit du Niger a╠Ç Ansongo.csv"))
    #         analyzer = PredictedToImpactPerformanceAnalyzer(cfg.DataDir, RPyr, impact_csv, cfg.triggerProb, cfg.adminLevel, cfg.admPath, cfg.startYear, cfg.endYear, cfg.years, PTM_events_per_adm, comparisonType, cfg.actionLifetime, 'PTM', leadtime)
    #         analyzer.matchImpact_and_Trigger()
    #         analyzer.calculateCommunePerformance()
    
    # calculate observation for timeframe 
    comparisonType ='Observation'

    # for percentile in cfg.percentiles: 
    #     observation_per_station = pd.read_csv (f"{cfg.DataDir}/Observation/observationalStation_flood_events_percentile{percentile:.1f}.csv")

        

    #     # PTM_events_per_adm = events_per_adm (cfg.DataDir, cfg.admPath, cfg.adminLevel, cfg.DNHstations, cfg.stationsDir, ptm_events_df, 'PTM', RPyr)
    #     for leadtime in cfg.leadtimes:

    #         analyzerPTM = PredictedToImpactPerformanceAnalyzer(DataDir = cfg.DataDir, 
    #                                                             RPyr=percentile, 
    #                                                             impactData=observation_per_station, 
    #                                                             triggerProb=cfg.triggerProb,  
    #                                                             startYear = cfg.startYear, 
    #                                                             endYear = cfg.endYear, 
    #                                                             years=cfg.years, 
    #                                                             PredictedEvents_gdf = ptm_events_df, 
    #                                                             comparisonType = comparisonType, 
    #                                                             actionLifetime = cfg.actionLifetime, 
    #                                                             model = 'PTM', 
    #                                                             leadtime = leadtime,
    #                                                             stationcoordinates=cfg.DNHstations)
    #         analyzerPTM.matchImpact_and_Trigger()
    #         analyzerPTM.calculateCommunePerformance()     


    # for RPyr in cfg.RPsyr: 
    #     #okay so now return period, setting it low to a standard to make sure there are many hits 
    #     observation_per_station = pd.read_csv(f'{cfg.DataDir}/{comparisonType}/observationalStation_flood_events_RP_{RPyr:.1f}yr.csv')
    #     ptm_events_df = ptm_events (cfg.DNHstations, cfg.DataDir, 'RP', RPyr, cfg.StationCombos)

    #     # PTM_events_per_adm = events_per_adm (cfg.DataDir, cfg.admPath, cfg.adminLevel, cfg.DNHstations, cfg.stationsDir, ptm_events_df, 'PTM', RPyr)
    #     for leadtime in cfg.leadtimes:
    #         analyzerPTM = PredictedToImpactPerformanceAnalyzer(DataDir = cfg.DataDir, 
    #                                                             RPyr=RPyr, 
    #                                                             impactData=observation_per_station, 
    #                                                             triggerProb=cfg.triggerProb,  
    #                                                             startYear = cfg.startYear, 
    #                                                             endYear = cfg.endYear, 
    #                                                             years=cfg.years, 
    #                                                             PredictedEvents_gdf = ptm_events_df, 
    #                                                             comparisonType = comparisonType, 
    #                                                             actionLifetime = cfg.actionLifetime, 
    #                                                             model = 'PTM', 
    #                                                             leadtime = leadtime,
    #                                                             stationcoordinates=cfg.DNHstations)
    #         analyzerPTM.matchImpact_and_Trigger()
    #         analyzerPTM.calculateCommunePerformance()   
    
    # GloFAS: 
    for RPyr in cfg.RPsyr: 
        observation_per_station = pd.read_csv (f"{cfg.DataDir}/Observation/observationalStation_flood_events_RP_{RPyr:.1f}yr.csv")
        for leadtime in cfg.leadtimes:
            glofas3_station_events = str(cfg.DataDir / f"GloFAS_31_rfcst_Mali/GloFASstation_flood_events_RP{RPyr:.1f}yr_leadtime{leadtime:.1f}.csv")
            analyzerG = PredictedToImpactPerformanceAnalyzer(DataDir = cfg.DataDir, 
                                                                RPyr=RPyr, 
                                                                impactData=observation_per_station, 
                                                                triggerProb=cfg.triggerProb,  
                                                                startYear =cfg.startYear, # for compatibility with glofas
                                                                endYear= cfg.endYear, 
                                                                years=cfg.years, 
                                                                PredictedEvents_gdf = glofas3_station_events, 
                                                                comparisonType = comparisonType, 
                                                                actionLifetime = cfg.actionLifetime, 
                                                                model = 'GloFAS3', 
                                                                leadtime = leadtime,
                                                                stationcoordinates=cfg.DNHstations)
            analyzerG.matchImpact_and_Trigger()
            analyzerG.calculateCommunePerformance() 

    for percentile in cfg.percentiles: 
        observation_per_station = pd.read_csv (f"{cfg.DataDir}/Observation/observationalStation_flood_events_percentile{percentile:.1f}.csv")
        for leadtime in cfg.leadtimes:
            glofas3_station_events = str(cfg.DataDir / f"GloFAS_31_rfcst_Mali/GloFASstation_flood_events_RP{RPyr:.1f}yr_leadtime{leadtime:.1f}.csv")
            analyzerG = PredictedToImpactPerformanceAnalyzer(DataDir = cfg.DataDir, 
                                                                RPyr=percentile, 
                                                                impactData=observation_per_station, 
                                                                triggerProb=cfg.triggerProb,  
                                                                startYear =cfg.startYear, 
                                                                endYear= cfg.endYear, 
                                                                years=cfg.years, 
                                                                PredictedEvents_gdf = glofas3_station_events, 
                                                                comparisonType = comparisonType, 
                                                                actionLifetime = cfg.actionLifetime, 
                                                                model = 'GloFAS3', 
                                                                leadtime = leadtime,
                                                                stationcoordinates=cfg.DNHstations)
            analyzerG.matchImpact_and_Trigger()
            analyzerG.calculateCommunePerformance() 


    # comparisonType ='Impact'
    # impact_per_admin = pd.read_csv (f"{cfg.DataDir}/{comparisonType}_data/impact_events_per_admin_529.csv", delimiter=';')

    # for percentile in cfg.percentiles: 
    #     ptm_events_df = ptm_events (cfg.DNHstations, cfg.DataDir, 'percentile', percentile, cfg.StationCombos)
    #     PTM_events_per_adm = events_per_adm (cfg.DataDir, cfg.admPath, cfg.adminLevel, cfg.DNHstations, cfg.stationsDir, ptm_events_df, 'PTM', RPyr)
    #     for leadtime in cfg.leadtimes:
    #         analyzerPTM = PredictedToImpactPerformanceAnalyzer(DataDir = cfg.DataDir, 
    #                                                             RPyr=percentile, 
    #                                                             impactData=impact_per_admin, 
    #                                                             triggerProb=cfg.triggerProb,  
    #                                                             startYear = cfg.startYear, 
    #                                                             endYear = cfg.endYear, 
    #                                                             years=cfg.years, 
    #                                                             PredictedEvents_gdf = PTM_events_per_adm, 
    #                                                             comparisonType = comparisonType, 
    #                                                             actionLifetime = cfg.actionLifetime, 
    #                                                             model = 'PTM', 
    #                                                             leadtime = leadtime,
    #                                                             adminLevel=cfg.adminLevel, 
    #                                                             adminPath=cfg.admPath,
    #                                                             stationcoordinates=None)
                                                                
    #         analyzerPTM.matchImpact_and_Trigger()
    #         analyzerPTM.calculateCommunePerformance()     


    # for RPyr in cfg.RPsyr: 
    #      #okay so now return period, setting it low to a standard to make sure there are many hits 
    #     ptm_events_df = ptm_events (cfg.DNHstations, cfg.DataDir, 'RP', RPyr, cfg.StationCombos)
    #     PTM_events_per_adm = events_per_adm (cfg.DataDir, cfg.admPath, cfg.adminLevel, cfg.DNHstations, cfg.stationsDir, ptm_events_df, 'PTM', RPyr)
    #     for leadtime in cfg.leadtimes:
    #         analyzerPTM = PredictedToImpactPerformanceAnalyzer(DataDir = cfg.DataDir, 
    #                                                             RPyr=RPyr, 
    #                                                             impactData=impact_per_admin, 
    #                                                             triggerProb=cfg.triggerProb,  
    #                                                             startYear = cfg.startYear, 
    #                                                             endYear = cfg.endYear, 
    #                                                             years=cfg.years, 
    #                                                             PredictedEvents_gdf = PTM_events_per_adm, 
    #                                                             comparisonType = comparisonType, 
    #                                                             actionLifetime = cfg.actionLifetime, 
    #                                                             model = 'PTM', 
    #                                                             leadtime = leadtime,
    #                                                             adminLevel=cfg.adminLevel,
    #                                                             adminPath=cfg.admPath,
    #                                                             stationcoordinates=None)
    #         analyzerPTM.matchImpact_and_Trigger()
    #         analyzerPTM.calculateCommunePerformance()   
    
    # GloFAS: 
    # model = 'GloFAS'
    # for RPyr in cfg.RPsyr: 
    #     for leadtime in cfg.leadtimes:
    #         glofas_admin_prob= str(cfg.DataDir / f"{model}/floodedRP{RPyr:.1f}yr_leadtime{leadtime}_ADM{cfg.adminLevel}.gpkg")
    #         fd = FloodDefiner(cfg.adminLevel)
    #         glofas_admin_events = fd.EventMaker(glofas_admin_prob, cfg.actionLifetime, cfg.triggerProb)
    #         analyzerG = PredictedToImpactPerformanceAnalyzer(DataDir = cfg.DataDir, 
    #                                                             RPyr=RPyr, 
    #                                                             impactData=impact_per_admin, 
    #                                                             triggerProb=cfg.triggerProb,  
    #                                                             startYear =cfg.startYear, # for compatibility with glofas
    #                                                             endYear= cfg.endYear, 
    #                                                             years=cfg.years, 
    #                                                             PredictedEvents_gdf = glofas_admin_events, 
    #                                                             comparisonType = comparisonType, 
    #                                                             actionLifetime = cfg.actionLifetime, 
    #                                                             model = model, 
    #                                                             leadtime = leadtime,
    #                                                             adminLevel=cfg.adminLevel,
    #                                                             adminPath=cfg.admPath,
    #                                                             stationcoordinates=None)
    #         analyzerG.matchImpact_and_Trigger()
    #         analyzerG.calculateCommunePerformance() 

    # for percentile in cfg.percentiles: 
    #     observation_per_station = pd.read_csv (f"{cfg.DataDir}/Observation/observationalStation_flood_events_percentile{percentile:.1f}.csv")
    #     for leadtime in cfg.leadtimes:
    #         glofas_station_events = str(cfg.stationsDir / f"GloFAS_Q/GloFASstation_flood_events_percentile{percentile:.1f}_leadtime{leadtime}.csv")
    #         analyzerG = PredictedToImpactPerformanceAnalyzer(DataDir = cfg.DataDir, 
    #                                                             RPyr=percentile, 
    #                                                             impactData=observation_per_station, 
    #                                                             triggerProb=cfg.triggerProb,  
    #                                                             startYear =cfg.startYear, 
    #                                                             endYear= cfg.endYear, 
    #                                                             years=cfg.years, 
    #                                                             PredictedEvents_gdf = glofas_station_events, 
    #                                                             comparisonType = comparisonType, 
    #                                                             actionLifetime = cfg.actionLifetime, 
    #                                                             model = 'GloFAS', 
    #                                                             leadtime = leadtime,
    #                                                             stationcoordinates=cfg.DNHstations)
    #         analyzerG.matchImpact_and_Trigger()
    #         analyzerG.calculateCommunePerformance() 