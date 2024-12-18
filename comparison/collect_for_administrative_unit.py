import os
import pandas as pd
import numpy as np

def collect_performance_measures_over_station(StationName, DataDir, leadtimes, return_periods, percentiles):
    """
    Collects performance metrics (POD, FAR) for a given administrative unit and organizes them into arrays
    for both return periods and percentiles.

    Parameters:
    - admin_unit (str): Name of the administrative unit (ADM2 field in the CSVs).
    - DataDir (str): Base directory containing subdirectories for models (GloFAS, Google Floodhub, PTM).
    - leadtimes (list): List of leadtimes to include (defines one axis of the array).
    - return_periods (list): List of return periods (defines thresholds for one type of analysis).
    - percentiles (list): List of percentiles (defines thresholds for another type of analysis).

    Returns:
    - data (dict): Nested dictionary containing performance metrics organized by leadtime and threshold type.
    """
    models = ['GloFAS', 'GoogleFloodHub', 'PTM']
    data = {
        'leadtimes': leadtimes,
        'thresholds': {
            'return_periods': return_periods,
            'percentiles': percentiles
        },
        'POD': {model: {
                    'Observation': {
                        'return_periods': np.full((len(return_periods), len(leadtimes)), np.nan),
                        'percentiles': np.full((len(percentiles), len(leadtimes)), np.nan)
                    }
                } for model in models},
        'FAR': {model: {
                    'Observation': {
                        'return_periods': np.full((len(return_periods), len(leadtimes)), np.nan),
                        'percentiles': np.full((len(percentiles), len(leadtimes)), np.nan)
                    }
                } for model in models}
    }
    
    for model in models:
        model_dir = os.path.join(DataDir, model.replace(' ', ''))  # Adjust directory naming if needed
        if not os.path.exists(model_dir):
            print(f"Directory not found for model: {model}, skipping.")
            continue
        
        for comparison_type in ['Observation']:# 'Impact']:
            comp_dir = os.path.join(model_dir, comparison_type)
            if not os.path.exists(comp_dir):
                print(f"Directory not found for comparison type: {comparison_type} in {model}, skipping.")
                continue
            
            # Collect data for return periods
            for rp_idx, return_period in enumerate(return_periods):
                for lt_idx, leadtime in enumerate(leadtimes):
                    file_name = f'scores_byCommuneRP{return_period:.1f}_yr_leadtime{leadtime}.csv'
                    file_path = os.path.join(comp_dir, file_name)
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            row = df[df['StationName'] == StationName]
                            if not row.empty:
                                pod_value = row['pod'].values[0] if 'pod' in row else np.nan
                                far_value = row['far'].values[0] if 'far' in row else np.nan
                                data['POD'][model][comparison_type]['return_periods'][rp_idx, lt_idx] = pod_value
                                data['FAR'][model][comparison_type]['return_periods'][rp_idx, lt_idx] = far_value
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}, skipping.")
            
            # Collect data for percentiles
            for pct_idx, percentile in enumerate(percentiles):
                for lt_idx, leadtime in enumerate(leadtimes):#so a bit weird format but ok:
                    file_name = f'scores_byCommuneRP{percentile:.1f}_yr_leadtime{leadtime}.csv'
                    file_path = os.path.join(comp_dir, file_name)
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            row = df[df['StationName'] == StationName]
                            if not row.empty:
                                pod_value = row['pod'].values[0] if 'pod' in row else np.nan
                                far_value = row['far'].values[0] if 'far' in row else np.nan
                                data['POD'][model][comparison_type]['percentiles'][pct_idx, lt_idx] = pod_value
                                data['FAR'][model][comparison_type]['percentiles'][pct_idx, lt_idx] = far_value
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}, skipping.")
    
    return data

def collect_performance_measures_over_admin(admin_unit, DataDir, leadtimes, return_periods, percentiles):
    """
    Collects performance metrics (POD, FAR) for a given administrative unit and organizes them into arrays
    for both return periods and percentiles.

    Parameters:
    - admin_unit (str): Name of the administrative unit (ADM2 field in the CSVs).
    - DataDir (str): Base directory containing subdirectories for models (GloFAS, Google Floodhub, PTM).
    - leadtimes (list): List of leadtimes to include (defines one axis of the array).
    - return_periods (list): List of return periods (defines thresholds for one type of analysis).
    - percentiles (list): List of percentiles (defines thresholds for another type of analysis).

    Returns:
    - data (dict): Nested dictionary containing performance metrics organized by leadtime and threshold type.
    """
    models = ['GloFAS', 'GoogleFloodHub', 'PTM']
    data = {
        'leadtimes': leadtimes,
        'thresholds': {
            'return_periods': return_periods,
            'percentiles': percentiles
        },
        'POD': {model: {
                    'Observation': {
                        'return_periods': np.full((len(return_periods), len(leadtimes)), np.nan),
                        'percentiles': np.full((len(percentiles), len(leadtimes)), np.nan)
                    },
                    'Impact': {
                        'return_periods': np.full((len(return_periods), len(leadtimes)), np.nan),
                        'percentiles': np.full((len(percentiles), len(leadtimes)), np.nan)
                    }
                } for model in models},
        'FAR': {model: {
                    'Observation': {
                        'return_periods': np.full((len(return_periods), len(leadtimes)), np.nan),
                        'percentiles': np.full((len(percentiles), len(leadtimes)), np.nan)
                    },
                    'Impact': {
                        'return_periods': np.full((len(return_periods), len(leadtimes)), np.nan),
                        'percentiles': np.full((len(percentiles), len(leadtimes)), np.nan)
                    }
                } for model in models}
    }
    
    for model in models:
        model_dir = os.path.join(DataDir, model.replace(' ', ''))  # Adjust directory naming if needed
        if not os.path.exists(model_dir):
            print(f"Directory not found for model: {model}, skipping.")
            continue
        
        for comparison_type in ['Observation', 'Impact']:
            comp_dir = os.path.join(model_dir, comparison_type)
            if not os.path.exists(comp_dir):
                print(f"Directory not found for comparison type: {comparison_type} in {model}, skipping.")
                continue
            
            # Collect data for return periods
            for rp_idx, return_period in enumerate(return_periods):
                for lt_idx, leadtime in enumerate(leadtimes):
                    file_name = f'scores_byCommuneRP{return_period}_yr_leadtime{leadtime}.csv'
                    file_path = os.path.join(comp_dir, file_name)
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            row = df[df['ADM2'] == admin_unit]
                            if not row.empty:
                                pod_value = row['pod'].values[0] if 'pod' in row else np.nan
                                far_value = row['far'].values[0] if 'far' in row else np.nan
                                data['POD'][model][comparison_type]['return_periods'][rp_idx, lt_idx] = pod_value
                                data['FAR'][model][comparison_type]['return_periods'][rp_idx, lt_idx] = far_value
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}, skipping.")
            
            # Collect data for percentiles
            for pct_idx, percentile in enumerate(percentiles):
                for lt_idx, leadtime in enumerate(leadtimes):#so a bit weird format but ok
                    file_name = f'scores_byCommuneRP{percentile}_yr_leadtime{leadtime}.csv'
                    file_path = os.path.join(comp_dir, file_name)
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            row = df[df['ADM2'] == admin_unit]
                            if not row.empty:
                                pod_value = row['pod'].values[0] if 'pod' in row else np.nan
                                far_value = row['far'].values[0] if 'far' in row else np.nan
                                data['POD'][model][comparison_type]['percentiles'][pct_idx, lt_idx] = pod_value
                                data['FAR'][model][comparison_type]['percentiles'][pct_idx, lt_idx] = far_value
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}, skipping.")
    
    return data



if __name__ =='__main__': 
    # These regions include Bamako, Koulikoro, Ségou, Mopti, Timbouctou and Gao, which have historically experienced frequent and severe flood events. 
    # Regions such as Mopti and Ségou are of particular concern due to their high exposure to flooding as well as their dense populations,
    # Bla is in Segou, impact data recorded there, no obs
    # Segou is in Segou, Observation data recorded  ,  only false negatives, but there is impact
    # Kolondieba in Sikasso, okay score in obs, no impact 
    # San in segou: impact data

    GloFASdata = collect_performance_measures_over_station('BAMAKO', cfg.DataDir, cfg.leadtimes, cfg.RPsyr, cfg.percentiles)
    print (GloFASdata)