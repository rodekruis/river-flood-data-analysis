
import GloFAS.GloFAS_prep.configuration as cfg
import pandas as pd
from GloFAS.GloFAS_prep.text_formatter import remove_accents, capitalize
import scipy.stats as stats
from comparison.pointMatching import attributePoints_to_Polygon
import geopandas as gpd
import numpy as np
from scipy.stats import genextreme

def parse_date_with_fallback(date_str, year):
    try:
        # Try to parse the date with the given year
        date = pd.to_datetime(f"{year} {date_str}", format="%Y %d/%m %H:%M")
        return date
    except ValueError:
        # If the date is invalid (e.g., Feb 29 in non-leap years), return None
        #print(f"Invalid date skipped: {date_str} for year {year}")
        return None
        
def transform_hydro(csvPath): 
    
    hydro_df_wide = pd.read_csv(csvPath, header=0)

    # Melt the wide-format DataFrame to long format
    hydro_df_long = hydro_df_wide.melt(id_vars=["Date"], var_name="Year", value_name="Value")

    # Apply the parse_date_with_fallback function row by row to create valid datetime objects
    def create_datetime(row):
        return parse_date_with_fallback(row['Date'], row['Year'])

    hydro_df_long['ParsedDate'] = hydro_df_long.apply(create_datetime, axis=1)

    # Drop rows where ParsedDate is None (invalid dates)
    hydro_df_long = hydro_df_long.dropna(subset=["ParsedDate"])

    # Drop the now redundant 'Year' and 'Date' columns if not needed
    hydro_df_long = hydro_df_long.drop(columns=["Year", "Date"])

    # Rename ParsedDate back to Date
    hydro_df_long.rename(columns={"ParsedDate": "Date"}, inplace=True)

    # Display the reshaped DataFrame
    return hydro_df_long

def z_RP_station(HydroStations_RP_file, StationName, RP):
    '''waterlevel
    '''
    HydroStations_RP_df = pd.read_csv(HydroStations_RP_file)
    
    # Remove accents from StationName
    StationName = remove_accents(StationName)
    
    # Handle NaN values and ensure all are strings before applying the function
    HydroStations_RP_df['StationName'] = HydroStations_RP_df['StationName'].fillna("").astype(str).apply(remove_accents)
    
    # Filter or calculate QRP based on RP and StationName
    z_RP = HydroStations_RP_df[HydroStations_RP_df['StationName'] == StationName][f'{RP}'].values

    return z_RP

def QRP_fit (hydro_df, RP): 

    # Extract the annual maximum discharge values
    hydro_df['Year'] = hydro_df['Date'].dt.year
    annual_max_discharge = hydro_df.groupby('Year')['Value'].max()

    # Fit a Gumbel distribution to the annual maximum discharge values
    loc, scale = stats.gumbel_r.fit(annual_max_discharge)
    # Calculate the discharge value corresponding to the return period
    discharge_value = stats.gumbel_r.ppf(1 - 1/RP, loc, scale)
    return discharge_value

def Q_GEV_fit(hydro_df, percentile): 
    """
    Fits a GEV distribution to the daily discharge values and calculates the discharge 
    corresponding to a given percentile.

    Parameters:
        hydro_df (pd.DataFrame): A dataframe with at least 'Date' and 'Value' columns.
            'Date' should be a datetime object and 'Value' is the discharge value.
        percentile (float): Percentile for which to compute the discharge value (between 0 and 100).

    Returns:
        float: The discharge value corresponding to the given percentile.
    """
    # Ensure 'Date' column is a datetime object
    hydro_df['Date'] = pd.to_datetime(hydro_df['Date'])

    # Extract daily discharge values
    daily_discharge = hydro_df['Value']
    #  Remove non-finite values
    daily_discharge_cleaned = daily_discharge[np.isfinite(daily_discharge)]

    # Check if there are still issues
    if daily_discharge_cleaned.empty:
        raise ValueError("All data was non-finite after cleaning. Please check your dataset.")

    # Fit a GEV distribution
    shape, loc, scale = genextreme.fit(daily_discharge_cleaned)


    # Calculate the discharge value corresponding to the specified percentile
    discharge_value = genextreme.ppf(percentile / 100, shape, loc=loc, scale=scale)

    return discharge_value


def stampHydroTrigger(hydro_df, RP, StationName): 
    """
    Adds a 'trigger' column to the hydro_df DataFrame indicating whether the 'Value' exceeds the QRP threshold.

    Parameters:
    - hydro_df (pd.DataFrame): DataFrame containing hydrological data with a 'Value' column.
    - RP (int/float): Return period for which the threshold is computed.
    - QRP_station (callable): Function that calculates QRP based on station data.
    - HydroStations_RP_file (str): File path for station return period data.
    - StationName (str): Name of the station for threshold calculation.

    Returns:
    - pd.DataFrame: Copy of the input DataFrame with an additional 'trigger' column.
    """
    # Ensure "Value" column exists in the DataFrame
    if "Value" not in hydro_df.columns:
        raise ValueError("The input DataFrame must contain a 'Value' column.")
    #QRP = QRP_station(HydroStations_RP_file, StationName, RP)
    Q_station = hydro_df["Value"] 
    
    if RP < 21:
        QRP= QRP_fit (hydro_df, RP)
    else: # assuming above 20 is percentile, RP is percentile instead 
        print ('applying a GEV, assuming your RP is in percentiles')
        QRP = Q_GEV_fit (hydro_df, RP) 
    #print (f"for {StationName} : return period Q= {QRP}")
    if not isinstance(QRP, (int, float)):
        raise ValueError(f"Expected QRP to be a scalar (int or float), but got {type(QRP)}.")
    # Calculate the QRP threshold
    

    # Copy the DataFrame and add the 'trigger' column
    hydro_trigger_df = hydro_df.copy()
    hydro_trigger_df['Trigger'] = (hydro_trigger_df['Value'] > QRP).astype(int)
    return hydro_trigger_df

def createEvent(trigger_df):
    # Sort the dataframe by 'ValidTime', then by administrative unit: so that for each administrative unit, they are 
    # first sort on valid time 
    trigger_df = trigger_df.sort_values(by=[f'Date']).reset_index(drop=True)

    # Prepare commune info with geometry for event creation
    event_data = []
            
    # Keep track of which rows have been processed to avoid rechecking
    processed_rows = [False] * len(trigger_df)
    
    r = 0
    while r < len(trigger_df):
        row = trigger_df.iloc[r]
        if processed_rows[r]:
            # If the row is already processed, continue to the next row
            r += 1
            continue
        
        # Checking if the current row and next row are both part of an event (Trigger == 1), and the trigger happens in the same adiministrative unit
        elif row['Trigger'] == 1 and r + 1 < len(trigger_df) and trigger_df.iloc[r + 1]['Trigger'] == 1:
            Event = 1
            StartDate = row['Date'] 
            # Continue until the end of the event where 'Trigger' is no longer 1
            while r < len(trigger_df) and trigger_df.iloc[r]['Trigger'] == 1:
                possible_endtime = trigger_df.iloc[r]['Date']
                processed_rows[r] = True  # Mark row as processed
                row = trigger_df.iloc[r]
                r += 1
                # print(f"Marked row, {row[f'ADM{adminLevel}']}, {row['ValidTime']}, is processed")
                # print (r)
            
            # The final EndValidTime after the loop finishes
            final_endtime = possible_endtime
                
            # Create a temporary dataframe for the current event
            temp_event_df = pd.DataFrame({
                'Observation': [Event],
                'Start Date': [StartDate],
                'End Date': [final_endtime],
            })
            
            # Append the event to the list of events
            event_data.append(temp_event_df)
                    
        else:
            # Move to the next row if no event is found
            r += 1

    # Concatenate all events into a single GeoDataFrame
    if event_data:
        # eventADM_data = pd.DataFrame(eventADM_data) 
        events_df = pd.concat(event_data, ignore_index=True)
        return events_df
    else:
        # Return an empty GeoDataFrame if no events were found
        # Initialize an empty dataframe 
        events_df = pd.DataFrame(columns=['Observation', 'Start Date', 'End Date'])
        return events_df

def loop_over_stations(station_csv, DataDir, RP): 
    RP = float(RP)
    station_df = pd.read_csv (station_csv, header=0)
    #print (station_df.columns)
    hydrodir = rf"{DataDir}/DNHMali_2019\Q_stations"
    all_events = []
    for _, stationrow in station_df.iterrows(): 
        StationName = stationrow['StationName']
        BasinName= stationrow['Basin']
        stationPath = rf"{hydrodir}/{BasinName}_{StationName}.csv"
        try: 
            hydro_df = transform_hydro (stationPath)
            print (f'calculating {StationName}, in {BasinName}')
        except: 
            # print (f'no discharge measures found for station {StationName} in {BasinName}')
            continue

        trigger_df = stampHydroTrigger (hydro_df, RP, StationName)
        event_df = createEvent (trigger_df)
        event_df ['StationName'] = StationName
        all_events.append (event_df)
    
    #generate the gdf to merge with where the points are attributed to the respective administrative units
    all_events_df = pd.concat (all_events, ignore_index=True)
    all_events_df.to_csv (f"{DataDir}/Observation/observationalStation_flood_events_RP_{RP}yr.csv")
    return all_events_df 

def events_per_adm (DataDir, admPath, adminLevel, station_csv, StationDataDir, all_events_df, model, RP):
    '''model may also be: observation / impact , its just to describe what type of representation they are for reality to store the data eventually'''
    gdf_pointPolygon = attributePoints_to_Polygon (admPath, station_csv, 'StationName', buffer_distance_meters=5000, StationDataDir=StationDataDir)
    gdf_pointPolygon.rename(columns={f'ADM{adminLevel}_FR':f'ADM{adminLevel}'}, inplace=True)
    gdf_pointPolygon [f'ADM{adminLevel}'] = gdf_pointPolygon [f'ADM{adminLevel}'].apply(capitalize)
    gdf_melt = gdf_pointPolygon.melt(
        id_vars=gdf_pointPolygon.columns.difference(['StationName_1', 'StationName_2', 'StationName_3', 'StationName_4']),
        value_vars=['StationName_1', 'StationName_2', 'StationName_3', 'StationName_4'],
        var_name='StationName_Type',  # Temporary column indicating the source column
        value_name='StationName_Merged'  # Use a unique column name here
        )
    gdf_melt = gdf_melt.dropna(subset=['StationName_Merged'])
    gdf_melt = gdf_melt.drop(columns='geometry')#.to_csv (f"{DataDir}/observation/adm_flood_events_RP{RP}yr.csv")
    # Proceed with the merge
    hydro_events_df = pd.merge(gdf_melt, all_events_df, left_on='StationName_Merged', right_on='StationName', how='inner')
    hydro_events_df.to_csv (f"{DataDir}/{model}/floodevents_admUnit_RP{RP}yr.csv")
    #hydro_events_gdf = gpd.GeoDataFrame(hydro_events_df, geometry='geometry')   
    #hydro_events_gdf.to_file(f"{DataDir}/observation/observational_flood_events_RP_{RP}yr.gpkg")
    #hydro_events_gdf.to_file
    return hydro_events_df


if __name__=="__main__": 

    #print (readcsv(f"{DataDir}/Données partagées - DNH Mali - 2019/Donne╠ües partage╠ües - DNH Mali - 2019/De╠übit du Niger a╠Ç Ansongo.csv"))
    event_df = loop_over_stations (cfg.DNHstations, cfg.DataDir, 1)
    event_gdf = events_per_adm (cfg.DataDir, cfg.admPath, cfg.adminLevel, cfg.DNHstations, cfg.stationsDir, event_df, 'Observation', 1)