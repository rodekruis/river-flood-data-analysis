from comparison.observation.HydroImpact import loop_over_stations_obs, events_per_adm
import pandas as pd 
from datetime import datetime, timedelta
import GloFAS.GloFAS_prep.configuration as cfg
columnID = 'StationName'
# k = [upstream, downstream], v = days of propagation time 


def ptm_events(DNHstations, DataDir, thresholdtype, threshold_value, StationCombos):
    # Generate station events data
    station_events_df = loop_over_stations_obs(DNHstations, DataDir, f'{thresholdtype}',threshold_value,value_col='Value')
    pred_ptm_events = []
    
    # Iterate through station combinations
    for combo in StationCombos:
        upstream_station = combo["Upstream"]
        downstream_station = combo["Downstream"]
        propagationtime = combo["PropagationTime"]
        
        for _, station_event in station_events_df.iterrows():
            if station_event['StationName'] == upstream_station:
                # Adjust dates by propagation time
                StartDate = pd.to_datetime(station_event['Start Date'], errors='coerce') + timedelta(days=propagationtime)
                EndDate = pd.to_datetime(station_event['End Date'], errors='coerce') + timedelta(days=propagationtime)
                
                # Create temporary DataFrame for the downstream event
                temp_one_station_one_event_df = pd.DataFrame({
                    'StationName': [downstream_station],
                    'Start Date': [StartDate],
                    'End Date': [EndDate],
                    'LeadTime': [propagationtime*24]
                })
                pred_ptm_events.append(temp_one_station_one_event_df)
    
    # Concatenate all predicted events into a single DataFrame
    if pred_ptm_events:
        ptm_events_df = pd.concat(pred_ptm_events, ignore_index=True)
    else:
        ptm_events_df = pd.DataFrame(columns=['StationName', 'Start Date', 'End Date', 'LeadTime'])
    
    return ptm_events_df


if __name__=='__main__': 
    StationCombos = [
        {"Upstream": "Banankoro", "Downstream": "Bamako", "PropagationTime": 4},
        {"Upstream": "Bamako", "Downstream": "Koulikoro", "PropagationTime": 1},
        {"Upstream": "Koulikoro", "Downstream": "Tamani", "PropagationTime": 3},
        {"Upstream": "Tamani", "Downstream": "Kirango", "PropagationTime": 2},
        {"Upstream": "Kirango", "Downstream": "Ké-Macina", "PropagationTime": 1},
    ]

    for RP in cfg.RPsyr: 
        ptm_events_df = ptm_events (cfg.DNHstations, cfg.DataDir, RP, StationCombos)
        PTM_events_per_adm = events_per_adm(cfg.DataDir, cfg.admPath, cfg.adminLevel, cfg.DNHstations, cfg.stationsDir, ptm_events_df, 'PTM', RP)

    