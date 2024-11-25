import pandas as pd
import geopandas as gpd
import numpy as np
import unidecode
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from GloFAS.GloFAS_analysis.flood_definer import FloodDefiner

from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
import GloFAS.GloFAS_prep.configuration as cfg
class PredictedToImpactPerformanceAnalyzer:
    def __init__(self, DataDir, RPyr, leadtime, impactData, triggerProb, adminLevel, adminPath, startYear, endYear, years, PredictedEvents_gdf, comparisonType):
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
        self.gdf_shape = gpd.read_file(adminPath)
        self.DataDir = DataDir 
        self.RPyr = RPyr 
        self.leadtime = leadtime
        self.startYear = startYear
        self.endYear = endYear
        self.years = years
        self.impactData = impactData
        self.PredictedEvents_gdf = PredictedEvents_gdf
        self.comparisonType = comparisonType
        
        if comparisonType =='Observation': 
            self.impact_gdf = self.openObservation_gdf()
        elif comparisonType =='Impact': 
            self.impact_gdf = self.openObservedImpact_gdf()
        else: 
            raise ValueError(f'no such comparisonType: {comparisonType}')
         # days before and after validtime the prediction is also valid

        
    def openObservation_gdf(self):
        # Load the data
        if self.impactData.endswith('.csv'):
            df = pd.read_csv(self.impactData)
        else: 
            df = gpd.read_file(self.impactData)

   
        # Remove time and convert 'Start Date' and 'End Date' to datetime
        df['Start Date'] = df['Start Date'].astype(str)
        df['End Date'] = df['End Date'].astype(str)

        # Remove the time by splitting at space , just keep the date as it is average mean nayway
        df['Start Date'] = df['Start Date'].str.split(' ', expand=True)[0]
        df['End Date'] = df['End Date'].str.split(' ', expand=True)[0]

        # Convert to datetime without specifying format to auto-detect
        df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
        df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')

        # Check the first few rows
        # Filter rows between 2004 and 2022 ()
        df_filtered = df[(df['End Date'].dt.year >= self.startYear) & (df['End Date'].dt.year < self.endYear)]
        # print (df_filtered.head)
        # Remove non-string entries from ADM columns
        df_filtered = df_filtered[df_filtered[f'ADM{self.adminLevel}'].apply(lambda x: isinstance(x, str))]
        self.gdf_shape = self.gdf_shape[self.gdf_shape[f'ADM{self.adminLevel}_FR'].apply(lambda x: isinstance(x, str))]

        #rename shapefile and make capital
        self.gdf_shape.rename(columns={f'ADM{cfg.adminLevel}_FR':f'ADM{cfg.adminLevel}'}, inplace=True)
        self.gdf_shape[f'ADM{cfg.adminLevel}'] = self.gdf_shape[f'ADM{cfg.adminLevel}'].apply(lambda x: unidecode.unidecode(x).upper())
        # Apply normalization to both DataFrames (converting to uppercase and removing special characters)
        
        df_filtered[f'ADM{self.adminLevel}'] = df_filtered[f'ADM{self.adminLevel}'].apply(lambda x: unidecode.unidecode(x).upper())
        
        # Merge the CSV data with the shapefile data
        impact_gdf = pd.merge(self.gdf_shape, df_filtered, how='left', left_on=f'ADM{cfg.adminLevel}', right_on=f'ADM{cfg.adminLevel}')
        
        return impact_gdf 


    def openObservedImpact_gdf(self):
        # Load the data
        if self.impactData.endswith('.csv'):
            df = pd.read_csv(self.impactData)
        else: 
            df = gpd.read_file(self.impactData)

        # Convert 'End Date' and 'Start Date' to datetime
         # still works, so after this wrong filtering begins 
        df['End Date'] = pd.to_datetime(df['End Date'], format='%d/%m/%Y', errors='coerce')
        df['Start Date'] = pd.to_datetime(df['Start Date'], format='%d/%m/%Y', errors= 'coerce')
        
        # Filter rows between 2004 and 2022 ()
        df_filtered = df[(df['End Date'].dt.year >= self.startYear) & (df['End Date'].dt.year < self.endYear)]
        # print (df_filtered.head)
        # Remove non-string entries from ADM columns
        df_filtered = df_filtered[df_filtered[f'ADM{self.adminLevel}'].apply(lambda x: isinstance(x, str))]
        self.gdf_shape = self.gdf_shape[self.gdf_shape[f'ADM{self.adminLevel}_FR'].apply(lambda x: isinstance(x, str))]

        #rename shapefile and make capital
        self.gdf_shape.rename(columns={f'ADM{cfg.adminLevel}_FR':f'ADM{cfg.adminLevel}'}, inplace=True)
        self.gdf_shape[f'ADM{cfg.adminLevel}'] = self.gdf_shape[f'ADM{cfg.adminLevel}'].apply(lambda x: unidecode.unidecode(x).upper())
        # Apply normalization to both DataFrames (converting to uppercase and removing special characters)
        
        df_filtered[f'ADM{self.adminLevel}'] = df_filtered[f'ADM{self.adminLevel}'].apply(lambda x: unidecode.unidecode(x).upper())
        
        # Merge the CSV data with the shapefile data
        impact_gdf = pd.merge(self.gdf_shape, df_filtered, how='left', left_on=f'ADM{cfg.adminLevel}', right_on=f'ADM{cfg.adminLevel}')
        
        return impact_gdf 


    def _check_ifmatched (self, commune, startdate, enddate):
        match = self.impact_gdf[
                        (self.impact_gdf[f'ADM{self.adminLevel}'] == commune) & 
                        (self.impact_gdf['Start Date'] <= enddate ) &
                        (self.impact_gdf['Start Date'] >= startdate )
                        ]
        return 1 if not match.empty else 0
    def clean_and_add_GloFAS (self, PredictedEvents_gdf): 
        """
        ONLY CLEAN ONCE MATCHES between impact-predicted HAVE BEEN MADE
        Clean GloFAS dataset by:
        - Removing entries where 'Event' is 0
        - Removing entries where there is no correspondence to the impact data (as to prevent double entries where matches have been made),
        - Removing entries for communes with no recorded impact in that year.
        After cleaning, add the remaining rows from GloFASCommune_gdf to impact_gdf,
        mapping 'StartValidTime' to 'Start Date', 'EndValidTime' to 'End Date', and
        f'ADM{self.adminLevel}' to f'ADM{self.adminLevel}'.
        """

        PredictedEvents_gdf['Remove'] = PredictedEvents_gdf.apply(
            lambda row: self._check_ifmatched(
                row[f'ADM{self.adminLevel}'], 
                row['StartValidTime'], 
                row['EndValidTime']),
                axis=1
            )

         # Keep entries from the original GloFASCommune_gdf where there was not yet a match
        PredictedEvents_gdf = PredictedEvents_gdf[PredictedEvents_gdf['Remove'] != 1] # remove was a 1, so we must keep everything that is not 1
        
        # NO YEAR IMPACT? --> no checking : this is the piece of code:
        # for year in self.years:
        #     # Convert year to date range for that year
        #     start_of_year = pd.to_datetime(f'{year}-01-01')
        #     end_of_year = pd.to_datetime(f'{year}-12-31')

        #     # Find communes that recorded an impact in the given year
        #     recorded_impacts = self.impact_gdf[
        #         (self.impact_gdf['Start Date'] >= start_of_year) &
        #         (self.impact_gdf['End Date'] <= end_of_year)
        #         ][f'ADM{self.adminLevel}'].unique()

        #     # Remove rows in GloFAS where no impact was recorded for that commune in that year
        #     PredictedEvents_gdf = PredictedEvents_gdf[
        #         ~(
        #             (PredictedEvents_gdf['StartValidTime'] >= start_of_year) &
        #             (PredictedEvents_gdf['EndValidTime'] <= end_of_year) &
        #             (~PredictedEvents_gdf[f'ADM{self.adminLevel}'].isin(recorded_impacts))
        #         )]

        # Prepare the remaining rows to be added to impact_gdf
        remaining_rows = PredictedEvents_gdf.copy()

        # Rename columns to match impact_gdf structure
        remaining_rows = remaining_rows.rename(
            columns={
                'StartValidTime': 'Start Date',
                'EndValidTime': 'End Date',
                f'ADM{self.adminLevel}': f'ADM{self.adminLevel}'
                })
        remaining_rows [f'{comparisonType}'] = 0 # we have established before that these are not matching to any impact data, so the impact at these remaining rows are 0 
        # Append the renamed rows to impact_gdf
        self.impact_gdf = pd.concat([self.impact_gdf, remaining_rows], ignore_index=True)

    def _check_impact(self, PredictedEvents_gdf, commune, startdate):
        '''Check if impact that has happened in the commune between given dates is RECORDED by glofas.'''
        match = PredictedEvents_gdf[
                                (PredictedEvents_gdf[f'ADM{self.adminLevel}'] == commune) & 
                                (PredictedEvents_gdf['StartValidTime'] <= startdate ) &
                                (PredictedEvents_gdf['EndValidTime'] >= startdate) &
                                (PredictedEvents_gdf['Event']==1)
                                ]
        return 1 if not match.empty else 0

    
    def matchImpact_and_Trigger(self):
        """
        Add 'Impact'  or observation and 'Event' columns to the floodCommune_gdf.
        - 'Impact': Whether the commune was impacted by a flood event.
        - 'Event': Whether the flood probability exceeds the defined threshold.
        """

        # Add Impact column using the check impact date (which only works on the impact gdf)
        
        self.impact_gdf['Event'] = self.impact_gdf.apply(
            lambda row: self._check_impact(self.PredictedEvents_gdf, row[f'ADM{self.adminLevel}'], row['Start Date']),
            axis=1
        )
        # Clean and add GloFAS to self.impact_gdf
        self.clean_and_add_GloFAS(self.PredictedEvents_gdf)
        # now add remaining GloFAS rows 
        # add rows for trigger entries = 1 , but no recorded impact 
        
        self.impact_gdf.to_file (f"{self.DataDir}/{comparisonType}/model_vs{comparisonType}RP{self.RPyr:.1f}_yr_leadtime{self.leadtime:.0f}.shp")
    
    def calc_performance_scores(self, obs, pred):
        '''Calculate performance scores based on observed and predicted values,
        including accuracy and precision, based on Google ML Crash Course definitions.'''

        # Define a DataFrame with consecutive classes and track hits, false alarms, misses, and correct negatives
        df = pd.DataFrame({'cons_class': pred.diff().ne(0).cumsum(), 'hits': (obs == 1) & (pred == 1)})
        hits = df[['cons_class', 'hits']].drop_duplicates().hits[df.hits == True].count()
        false_alarms = (pred.loc[pred.shift() != pred].sum()) - hits
        misses = sum((obs == 1) & (pred == 0))
        correct_negatives = sum((obs == 0) & (pred == 0))

        # Calculate metrics
        output = {
            'pod': hits / (hits + misses) if hits + misses != 0 else np.nan,  # Probability of Detection
            'far': false_alarms / (hits + false_alarms) if hits + false_alarms != 0 else np.nan,  # False Alarm Ratio
            'pofd': false_alarms / (false_alarms + correct_negatives) if false_alarms + correct_negatives != 0 else np.nan,  # Probability of False Detection
            'csi': hits / (hits + false_alarms + misses) if hits + false_alarms + misses != 0 else np.nan,  # Critical Success Index
            'accuracy': (hits + correct_negatives) / (hits + correct_negatives + false_alarms + misses) if hits + correct_negatives + false_alarms + misses != 0 else np.nan,  # Accuracy
            'precision': hits / (hits + false_alarms) if hits + false_alarms != 0 else np.nan  # Precision
        }

        return pd.Series(output)

    def calculateCommunePerformance(self):
        """
        Calculate the performance scores for each commune and merge them back into the GeoDataFrame.
        """
        # Group by 'Commune' and calculate performance scores for each group
        
        scores_by_commune = self.impact_gdf.groupby(f'ADM{self.adminLevel}').apply(
            lambda x: self.calc_performance_scores(x[f'{self.comparisonType}'], x['Event'])
        )
        scores_byCommune_gdf = self.gdf_shape.merge(scores_by_commune, on=f'ADM{cfg.adminLevel}')
        scores_byCommune_gdf.to_file (f"{self.DataDir}/{comparisonType}/scores_byCommuneRP{self.RPyr:.1f}_yr_leadtime{self.leadtime:.0f}.shp")
        return scores_byCommune_gdf

if __name__=='__main__':
    for RPyr in cfg.RPsyr: 
        comparisonType = 'Observation'
        hydro_impact_gdf = f'{cfg.DataDir}/{comparisonType}/observational_flood_events_RP_{RPyr}yr.csv'
        #hydro_impact_gdf = loop_over_stations (cfg.DNHstations , cfg.DataDir, RPyr, cfg.admPath, cfg.adminLevel)
        for leadtime in cfg.leadtimes: 
            floodProbability_path = cfg.DataDir/ f"floodedRP{RPyr}yr_leadtime{leadtime}_ADM{cfg.adminLevel}.gpkg"
            floodProbability_gdf = checkVectorFormat (floodProbability_path)
            #calculate the flood events
            definer = FloodDefiner (cfg.adminLevel)
            PredictedEvents_gdf = definer.EventMaker (floodProbability_gdf, cfg.actionLifetime, cfg.triggerProb)
        #print (readcsv(f"{DataDir}/Données partagées - DNH Mali - 2019/Donne╠ües partage╠ües - DNH Mali - 2019/De╠übit du Niger a╠Ç Ansongo.csv"))
            analyzer = PredictedToImpactPerformanceAnalyzer(cfg.DataDir, RPyr, leadtime, hydro_impact_gdf, cfg.triggerProb, cfg.adminLevel, cfg.admPath, cfg.startYear, cfg.endYear, cfg.years, PredictedEvents_gdf, comparisonType)
            analyzer.matchImpact_and_Trigger()
            analyzer.calculateCommunePerformance()

            