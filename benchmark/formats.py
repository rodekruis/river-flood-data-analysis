import xarray as xr
import numpy as np
import pandas as pd
import os 
import GloFAS.GloFAS_prep.config_comp as cfg
from comparison.observation.thresholds import Q_Gumbel_fit_percentile, Q_Gumbel_fit_RP
from comparison.observation.HydroImpact import createEvent
def get_stationcoordinates(path):
    stationsLon = []
    stationsLat = []
    stationsName = []
    with open(path, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("Lon"):  # Skip header
                parts = line.strip().split(";")
                stationsLon.append(float(parts[0]))  # Longitude
                stationsLat.append(float(parts[1]))  # Latitude
                stationsName.append (str(parts[3]))
    return stationsLon, stationsLat, stationsName

def path_generator (DataDir, stationLon, stationLat, forecastType = 'reforecast'): 
    if forecastType == 'reforecast':
        filepath = f"{DataDir}/GloFAS_31_rfcst_Mali/glofas_3_1_rfcst_pt_{stationLon}_{stationLat}_wts_1999_2018.nc"
    elif forecastType == 'reanalysis':
        filepath = f"{DataDir}/GloFAS31_threshold/glofas_3_1_rea_pt_{stationLon}_{stationLat}_wts_1979_2023.nc"
    else: 
        print ('No such forecastType')
        raise ValueError
    return filepath



def open_nc_and_save_csv_per_station(stationLon, stationLat, filepath, DataDir, 
                                forecastType = 'reanalysis', 
                                triggerProb=None,
                                leadtime=None):# in days):
    try:
        forecast = xr.load_dataset(filepath)  # Load NetCDF file
    except FileNotFoundError:
        print(f"File not found: {filepath}")

    if forecastType == 'reforecast':
        output_folder = f"{DataDir}/GloFAS_31_rfcst_Mali/"
        os.makedirs(output_folder, exist_ok=True)
        quantile = 1 - triggerProb 
        
        # Compute the 40th percentile along ensemble dimension
        percentile_40 = forecast['dis24'].quantile(quantile, dim="number")
   
        # Convert to Pandas DataFrame
        df = percentile_40.to_dataframe().reset_index()
        
        # Rename columns
        df.rename(columns={'valid_time': 'ValidTime', 'dis24': 'Discharge_40th_Percentile'}, inplace=True)
        
    elif forecastType =='reanalysis': 
        output_folder = f"{DataDir}/GloFAS31_threshold/"
        discharge = forecast ['dis24']
        df = discharge.to_dataframe().reset_index()
        df.rename(columns={'valid_time': 'ValidTime', 'dis24': 'Discharge'}, inplace=True)

        # Add station info
    df['Longitude'] = stationLon
    df['Latitude'] = stationLat

        # Define unique filename for each station
    output_csv = f"{output_folder}/discharge_{forecastType}_{stationLon}_{stationLat}.csv"

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV for station ({stationLon}, {stationLat}) to: {output_csv}")
    return df

def checking_columns (reforecast_df, date_col, reforecast_value_col):
    if date_col in reforecast_df.columns:
        # Convert the date_col to datetime
        reforecast_df[date_col] = pd.to_datetime(reforecast_df[date_col], format="%Y-%m-%d %H:%M", errors='coerce')

        # Check for any invalid dates
        if reforecast_df[date_col].isnull().any():
            print("Warning: Some dates in 'date_col' could not be parsed and are set to NaT.")
        
        # Set the index to the datetime column
        reforecast_df.set_index(date_col, inplace=True)
        #print(f"Index successfully set to datetime using '{date_col}':\n{hydro_df.index}")
    else:
        print(f"Warning: Column '{date_col}' does not exist in the DataFrame, hopefully the date is the index")

    # Ensure "Value" column exists in the DataFrame
    if reforecast_value_col not in reforecast_df.columns:
        raise ValueError("The input DataFrame must contain a 'Value' column, please specify the name correctly")
    return reforecast_df

def stampHydroTrigger(reforecast_df, reanalysis_df, type_of_extremity, probability, 
                        reforecast_value_col='Discharge_40th_Percentile', 
                        reanalysis_value_col='Discharge', 
                        date_col='ValidTime'): 
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

    reforecast_df = checking_columns (reforecast_df, date_col, reforecast_value_col)
    reanalysis_df = checking_columns (reanalysis_df, date_col, reanalysis_value_col)
    
    Q_station = reforecast_df[reforecast_value_col] 

    if type_of_extremity == 'RP':
        Q_prob= Q_Gumbel_fit_RP (reanalysis_df, probability, value_col=reanalysis_value_col, date_col=date_col) # results in infinite
        # Q prob is infinite
    elif type_of_extremity == 'percentile': # assuming above 20 is percentile, RP is percentile instead 
        Q_prob = Q_Gumbel_fit_percentile (reanalysis_df, probability, reanalysis_value_col, date_col= date_col) 
        
    else: 
        raise ValueError (f"no such type of extremity: {type_of_extremity}, pick 'percentile'or 'RP'")
    #print (f"for {StationName} : return period Q= {QRP}")
     # Q_prob is infinite so it makes sense that it is never exceeded ! 
    if not isinstance(Q_prob, (int, float)):
        raise ValueError(f"Expected QRP to be a scalar (int or float), but got {type(Q_prob)}.")
    # Calculate the QRP threshold
    # Copy the DataFrame and add the 'trigger' column
    hydro_trigger_df = reforecast_df.copy()
    hydro_trigger_df['Trigger'] = (hydro_trigger_df[reforecast_value_col] > Q_prob).astype(int)
    
    return hydro_trigger_df


def loop_over_stations_pred(coordinatespath, DataDir, probability, type_of_extremity, leadtime, date_col='ValidTime'): # leadtime in days): 
    probability = float(probability)
    stationsLon, stationsLat, stationsName = get_stationcoordinates(coordinatespath)
    all_events = []
    for stationLon, stationLat, stationName in zip(stationsLon, stationsLat, stationsName):
        print (stationName)
        rfcst_filepath = path_generator (DataDir, stationLon, stationLat, 'reforecast')
        reforecast_df = open_nc_and_save_csv_per_station(stationLon, stationLat, rfcst_filepath, DataDir, 'reforecast', cfg.triggerProb, leadtime)
        # NOne type ^
        rea_filepath = path_generator (DataDir, stationLon, stationLat, 'reanalysis')
        reanalysis_df = open_nc_and_save_csv_per_station(stationLon, stationLat, rea_filepath, DataDir, 'reanalysis', None, None)
  
        trigger_df = stampHydroTrigger (reforecast_df, reanalysis_df, type_of_extremity, probability)


        selected_lt_trigger_df = trigger_df[trigger_df['step']==f'{leadtime} days']
        event_df = createEvent (selected_lt_trigger_df, date_col)
        event_df ['StationName'] = stationName
        all_events.append (event_df)
        
    #generate the gdf to merge with where the points are attributed to the respective administrative units
    all_events_df = pd.concat (all_events, ignore_index=True)
    if type_of_extremity =='RP':
        all_events_df.to_csv (f"{DataDir}/GloFAS_31_rfcst_Mali/GloFASstation_flood_events_RP{probability}yr_leadtime{leadtime*24}.csv")
    elif type_of_extremity =='percentile': 
        all_events_df.to_csv (f"{DataDir}/GloFAS_31_rfcst_Mali/GloFASstation_flood_events_percentile{probability}_leadtime{leadtime*24}.csv")
    return all_events_df 
    
DataDir = cfg.DataDir
stationspath = f"{DataDir}/GloFAS_31_rfcst_Mali/Mali_Glo2.1_extracted_stations_DNH_AF_2021.txt"
for leadtime_hr in cfg.leadtimes:
    leadtime = leadtime_hr/24
    for RPyr in cfg.RPsyr:
        loop_over_stations_pred (stationspath, DataDir, RPyr, 'RP', leadtime)
    for percentile in cfg.percentiles:
        loop_over_stations_pred (stationspath, DataDir, percentile, 'percentile', leadtime)

    # threshold_ds = xr.load_dataset(DataDir / f"auxiliary/flood_threshold_glofas_v4_rl_{RPyr:.1f}.nc")
    # threshold_ds.rio.write_crs(crs, inplace=True)

    # threshold_da = threshold_ds[f"rl_{RPyr:.1f}"].sel(
    #     lat=slice(area[0], area[2]),
    #     lon=slice(area[1], area[3])
    # )
    # threshold_da = threshold_da.rename({'lat': 'latitude', 'lon': 'longitude'})
    #  #first getting just one Q_da of the first year to get the shape right
    # threshold_da = threshold_da.interp(latitude=Q_da_forecast.latitude, longitude=Q_da_forecast.longitude)
    # return threshold_da


