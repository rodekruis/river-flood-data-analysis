import pandas as pd
import GloFAS.GloFAS_prep.configuration as cfg
def readcsv(csvPath): 
    hydro_df_wide = pd.read_csv(csvPath, header=0)
    # have to make long format anyways eventually, might as well do it now 
    hydro_df_long = hydro_df_wide.melt(id_vars=["Date"], var_name="Year", value_name="Value") 
    # Combine 'Date' and 'Year' into a single datetime column
    print (hydro_df_long["Date"])
    hydro_df_long["Date"] = pd.to_datetime(hydro_df_long["Year"] + " " + hydro_df_long["Date"], format="%Y %d/%m %H:%M")

    # Drop the now redundant 'Year' column if not needed
    hydro_df_long = hydro_df_long.drop(columns=["Year"])

    # Display the reshaped DataFrame
    print(hydro_df_long)
    return hydro_df_long

def loop_over_stations(station_csv, DataDir): 
    station_df = pd.read_csv (station_csv, header=0)
    hydrodir = DataDir / f"Données partagées - DNH Mali - 2019/Donne╠ües partage╠ües - DNH Mali - 2019/"
    Q_station_df = pd.DataFrame()
    for _, stationrow in station_df.iterrows(): 
        
        StationName = stationrow['StationName']
        BasinName= stationrow['Basin']
        stationPath = hydrodir / f"De╠übits du {BasinName} a╠Ç {StationName}.csv"
        try: 
            hydro_df = readcsv (stationPath)
            Q_station_df [f'{StationName}'] = hydro_df [:,1]
        except: 
            print (f'no discharge measures found for station {StationName} in {BasinName}')
        
    return Q_station_df


def stampHydroTrigger(station, startYear, endYear): 
    
    return hydro_trigger_df

def makeImpact_gdf():
    impact_gdf['Start Date'] 
    impact_gdf['End Date']
    impact_gdf['Station']
    impact_gdf[f'ADM{adminLevel}']
    impact_gdf['Impact']
    return impact_gdf

if __name__=="__main__": 
    DataDir = cfg.DataDir
    station_csv = DataDir / f"Données partagées - DNH Mali - 2019/Stations_DNH.csv"
    print (readcsv(f"{DataDir}/Données partagées - DNH Mali - 2019/Donne╠ües partage╠ües - DNH Mali - 2019/De╠übit du Niger a╠Ç Ansongo.csv"))
    Q_station_df = loop_over_stations (station_csv, DataDir)
    print (Q_station_df)