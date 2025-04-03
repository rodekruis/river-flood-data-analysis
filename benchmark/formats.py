import xarray as xr
import numpy as np
import pandas as pd
import os 
import GloFAS.GloFAS_prep.config_comp as cfg
def get_stationcoordinates(path):
    stationsLon = []
    stationsLat = []

    with open(path, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("Lon"):  # Skip header
                parts = line.strip().split(";")
                stationsLon.append(float(parts[0]))  # Longitude
                stationsLat.append(float(parts[1]))  # Latitude
    return stationsLon, stationsLat


def open_nc_and_save_csv_per_station(coordinatespath, DataDir, output_folder):
    stationsLon, stationsLat = get_stationcoordinates(coordinatespath)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for stationLon, stationLat in zip(stationsLon, stationsLat):
        filepath = f"{DataDir}/GloFAS_31_rfcst_Mali/glofas_3_1_rfcst_pt_{stationLon}_{stationLat}_wts_1999_2018.nc"
        
        try:
            reforecast = xr.load_dataset(filepath)  # Load NetCDF file
            
            # Compute the 40th percentile along ensemble dimension
            percentile_40 = reforecast['dis24'].quantile(0.4, dim="number")

            # Convert to Pandas DataFrame
            df = percentile_40.to_dataframe().reset_index()

            # Rename columns
            df.rename(columns={'valid_time': 'ValidTime', 'dis24': 'Discharge_40th_Percentile'}, inplace=True)

            # Add station info
            df['Longitude'] = stationLon
            df['Latitude'] = stationLat

            # Define unique filename for each station
            output_csv = f"{output_folder}/reforecast_discharge_40th_percentile_{stationLon}_{stationLat}.csv"

            # Save CSV
            df.to_csv(output_csv, index=False)
            print(f"Saved CSV for station ({stationLon}, {stationLat}) to: {output_csv}")

        except FileNotFoundError:
            print(f"File not found: {filepath}")

            
def stampHydroTrigger(hydro_df, StationName, type_of_extremity, probability, value_col='Value', date_col='Date'): 
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
    # Assuming hydro_df is already loaded
    if date_col in hydro_df.columns:
        # Convert the date_col to datetime
        hydro_df[date_col] = pd.to_datetime(hydro_df[date_col], format="%Y-%m-%d %H:%M", errors='coerce')

        # Check for any invalid dates
        if hydro_df[date_col].isnull().any():
            print("Warning: Some dates in 'date_col' could not be parsed and are set to NaT.")
        
        # Set the index to the datetime column
        hydro_df.set_index(date_col, inplace=True)
        #print(f"Index successfully set to datetime using '{date_col}':\n{hydro_df.index}")
    else:
        print(f"Warning: Column '{date_col}' does not exist in the DataFrame, hopefully the date is the index")

    # Ensure "Value" column exists in the DataFrame
    if value_col not in hydro_df.columns:
        raise ValueError("The input DataFrame must contain a 'Value' column, please specify the name correctly")
    #QRP = QRP_station(HydroStations_RP_file, StationName, RP)
    Q_station = hydro_df[value_col] # this is where it goes wrong!!!
    if type_of_extremity == 'RP':
        Q_prob= Q_Gumbel_fit_RP (hydro_df, probability, value_col=value_col) # results in infinite
    elif type_of_extremity == 'percentile': # assuming above 20 is percentile, RP is percentile instead 
        Q_prob = Q_Gumbel_fit_percentile (hydro_df, probability, value_col) 
    else: 
        raise ValueError (f"no such type of extremity: {type_of_extremity}, pick 'percentile'or 'RP'")
    #print (f"for {StationName} : return period Q= {QRP}")
    if not isinstance(Q_prob, (int, float)):
        raise ValueError(f"Expected QRP to be a scalar (int or float), but got {type(Q_prob)}.")
    # Calculate the QRP threshold
    # Copy the DataFrame and add the 'trigger' column
    hydro_trigger_df = hydro_df.copy()
    hydro_trigger_df['Trigger'] = (hydro_trigger_df[value_col] > Q_prob).astype(int)
    return hydro_trigger_df
DataDir = cfg.DataDir
stationspath = f"{DataDir}/GloFAS_31_rfcst_Mali/Mali_Glo2.1_extracted_stations_DNH_AF_2021.txt"
output_folder = f"{DataDir}/GloFAS_31_rfcst_Mali/"
open_nc_and_save_csv_per_station(stationspath, DataDir, output_folder)

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


