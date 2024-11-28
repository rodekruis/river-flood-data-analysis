import os
import pandas as pd
import numpy as np

def collect_performance_measures(admin_unit, DataDir, leadtimes, return_periods):
    """
    Collects performance metrics (POD, FAR) for a given administrative unit and organizes them into 2D arrays.

    Parameters:
    - admin_unit (str): Name of the administrative unit (ADM2 field in the CSVs).
    - DataDir (str): Base directory containing subdirectories for models (GloFAS, Google Floodhub, EAP).
    - leadtimes (list): List of leadtimes to include (defines columns of the 2D array).
    - return_periods (list): List of return periods to include (defines rows of the 2D array).

    Returns:
    - data (dict): Nested dictionary containing performance metrics in 2D arrays:
      {
          'leadtimes': [...],
          'return_periods': [...],
          'POD': {
              'GloFAS': {'observation': np.array([...]), 'impact': np.array([...])},
              'Google Floodhub': {'observation': np.array([...]), 'impact': np.array([...])},
              'EAP': {'observation': np.array([...]), 'impact': np.array([...])}
          },
          'FAR': {
              'GloFAS': {'observation': np.array([...]), 'impact': np.array([...])},
              'Google Floodhub': {'observation': np.array([...]), 'impact': np.array([...])},
              'EAP': {'observation': np.array([...]), 'impact': np.array([...])}
          }
      }
    """
    models = ['GloFAS', 'GoogleFloodHub', 'EAP']
    data = {
        'leadtimes': leadtimes,
        'return_periods': return_periods,
        'POD': {model: {'observation': np.full((len(return_periods), len(leadtimes)), np.nan),
                        'impact': np.full((len(return_periods), len(leadtimes)), np.nan)} for model in models},
        'FAR': {model: {'observation': np.full((len(return_periods), len(leadtimes)), np.nan),
                        'impact': np.full((len(return_periods), len(leadtimes)), np.nan)} for model in models}
    }
    
    for model in models:
        model_dir = os.path.join(DataDir, model.replace(' ', ''))  # Adjust directory naming if needed
        if not os.path.exists(model_dir):
            print(f"Directory not found for model: {model}, skipping.")
            continue
        
        for comparison_type in ['observation', 'impact']:
            comp_dir = os.path.join(model_dir, comparison_type)
            if not os.path.exists(comp_dir):
                print(f"Directory not found for comparison type: {comparison_type} in {model}, skipping.")
                continue
            
            for i, return_period in enumerate(return_periods):  # Row index for return periods
                for j, leadtime in enumerate(leadtimes):  # Column index for lead times
                    # Build the filename
                    file_name = f'scores_byCommuneRP{return_period}_yr_leadtime{leadtime}.csv'
                    file_path = os.path.join(comp_dir, file_name)
                    
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}, skipping.")
                        continue
                    
                    # Read the CSV file
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}, skipping.")
                        continue
                    
                    # Filter rows for the specified admin unit
                    row = df[df['ADM2'] == admin_unit]
                    if row.empty:
                        print(f"Admin unit {admin_unit} not found in file: {file_path}, skipping.")
                    else:
                        # Extract POD and FAR values
                        pod_value = row['pod'].values[0] if 'pod' in row else np.nan
                        far_value = row['far'].values[0] if 'far' in row else np.nan
                        
                        data['POD'][model][comparison_type][i, j] = pod_value
                        data['FAR'][model][comparison_type][i, j] = far_value
    
    return data


if __name__ =='__main__': 
    # These regions include Bamako, Koulikoro, Ségou, Mopti, Timbouctou and Gao, which have historically experienced frequent and severe flood events. 
    # Regions such as Mopti and Ségou are of particular concern due to their high exposure to flooding as well as their dense populations,
    # Bla is in Segou, impact data recorded there, no obs
    # Segou is in Segou, observation data recorded  ,  only false negatives, but there is impact
    # Kolondieba in Sikasso, okay score in obs, no impact 
    # San in segou: impact data

    GloFASdata = collect_performance_measures('BLA', cfg.DataDir, cfg.leadtimes, cfg.RPsyr)
    print (GloFASdata)