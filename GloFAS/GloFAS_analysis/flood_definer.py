from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
import pandas as pd
import geopandas as gpd
import numpy as np
import unidecode
import configuration as cfg
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class FloodDefiner:
    def __init__(adminLevel):
        self.adminLevel = adminLevel

    def stampTrigger (self, floodProbability_gdf, actionLifetime, triggerProb):
        '''defines trigger by the floodprobability that it needs to exceed, as well as writes the timeduration at which the trigger is valid 
        actionLifetime = actionLifetime in days'''
        
        # make everything capital
        floodProbability_gdf[f'ADM{self.adminLevel}'] = floodProbability_gdf[f'ADM{self.adminLevel}'].apply(lambda x: unidecode.unidecode(x).upper())
        
        floodCommune_gdf['StartValidTime'] = pd.to_datetime(floodCommune_gdf['ValidTime'], errors='coerce') # - timedelta(days=self.buffervalid)
        # incorporate actionlifetime
        floodCommune_gdf['EndValidTime'] = pd.to_datetime(floodCommune_gdf['ValidTime'], errors='coerce') + timedelta(days=actionLifetime)
        floodCommune_gdf['StartValidTime'] = pd.to_datetime(floodCommune_gdf['StartValidTime'], format='%d/%m/%Y', errors='coerce')
        floodCommune_gdf['EndValidTime'] = pd.to_datetime(floodCommune_gdf['EndValidTime'], format='%d/%m/%Y', errors='coerce')
        # Add Trigger column based on probability threshold
        floodCommune_gdf['Trigger'] = np.where(floodCommune_gdf['Probability'] >= triggerProb, 1, 0)

        return floodCommune_gdf

    def createEvent(self, triggerstamped_gdf):
        # Sort the dataframe by 'ValidTime', then by administrative unit: so that for each administrative unit, they are 
        # First sort on administrative level,  then sort on valid time 
        triggerstamped_gdf = triggerstamped_gdf.sort_values(by=[f'ADM{adminLevel}','ValidTime']).reset_index(drop=True)

        # Prepare commune info with geometry for event creation
        commune_info_df = triggerstamped_gdf[[f'ADM{adminLevel}', 'geometry']].copy()
        
        eventADM_data = []
                
        # Keep track of which rows have been processed to avoid rechecking
        processed_rows = [False] * len(triggerstamped_gdf)
        
        r = 0
        while r < len(triggerstamped_gdf):
            row = triggerstamped_gdf.iloc[r]
            if processed_rows[r]:
                # If the row is already processed, continue to the next row
                r += 1
                continue
            
            # Checking if the current row and next row are both part of an event (Trigger == 1), and the trigger happens in the same adiministrative unit
            elif row['Trigger'] == 1 and r + 1 < len(triggerstamped_gdf) and triggerstamped_gdf.iloc[r + 1]['Trigger'] == 1 and row[f'ADM{adminLevel}']==triggerstamped_gdf.iloc[r + 1][f'ADM{adminLevel}']:
                Event = 1
                StartValidTime = row['StartValidTime'] 
                # Continue until the end of the event where 'Trigger' is no longer 1
                while r < len(triggerstamped_gdf) and triggerstamped_gdf.iloc[r]['Trigger'] == 1:
                    possible_endtime = triggerstamped_gdf.iloc[r]['EndValidTime']
                    processed_rows[r] = True  # Mark row as processed
                    r += 1
                    row = triggerstamped_gdf.iloc[r]
                    # print(f"Marked row, {row[f'ADM{adminLevel}']}, {row['ValidTime']}, is processed")
                    # print (r)
                
                # The final EndValidTime after the loop finishes
                final_endtime = possible_endtime
                    
                # Create a temporary dataframe for the current event
                temp_event_df = pd.DataFrame({
                    f'ADM{adminLevel}': [row[f'ADM{adminLevel}']],
                    'Event': [Event],
                    'StartValidTime': [StartValidTime],
                    'EndValidTime': [final_endtime],
                    'geometry': row['geometry']
                })
                
                # Append the event to the list of events
                eventADM_data.append(temp_event_df)
                        
            else:
                # Move to the next row if no event is found
                r += 1

        # Concatenate all events into a single GeoDataFrame
        if eventADM_data:
            # eventADM_data = pd.DataFrame(eventADM_data) 
            GloFASevents_gdf = pd.concat(eventADM_data, ignore_index=True)
            GloFASevents_gdf = gpd.GeoDataFrame(GloFASevents_gdf, geometry='geometry')

            return GloFASevents_gdf
        else:
            # Return an empty GeoDataFrame if no events were found
            # Initialize an empty dataframe 
            events_df = pd.DataFrame(columns=[f'ADM{adminLevel}', 'Event', 'StartValidTime', 'EndValidTime', 'geometry'])
            return gpd.GeoDataFrame(events_df, geometry='geometry')

    def EventMaker(self, floodProbability_gdf, actionLifetime, triggerProb): 
        triggerstamped_gdf = self.stampTrigger (floodProbability_gdf, actionLifetime, triggerProb)
        gdf = createEvent(triggerstamped_gdf)
        return gdf
