import os
import pandas as pd

def collect_performance_measures(admin_unit, DataDir, leadtimes, return_periods):
    """
    Collects performance metrics (POD, FAR) for a given administrative unit.

    Parameters:
    - admin_unit (str): Name of the administrative unit (ADM2 field in the CSVs).
    - DataDir (str): Base directory containing subdirectories for models (GloFAS, Google Floodhub, EAP).
    - leadtimes (list): List of leadtimes to include.
    - return_periods (list): List of return periods to include.

    Returns:
    - data (dict): Nested dictionary containing performance metrics:
      {
          'leadtimes': [...],
          'return_periods': [...],
          'POD': {
              'GloFAS': {'observation': [...], 'impact': [...]},
              'Google Floodhub': {'observation': [...], 'impact': [...]},
              'EAP': {'observation': [...], 'impact': [...]}
          },
          'FAR': {
              'GloFAS': {'observation': [...], 'impact': [...]},
              'Google Floodhub': {'observation': [...], 'impact': [...]},
              'EAP': {'observation': [...], 'impact': [...]}
          }
      }
    """
    models = ['GloFAS', 'Google Floodhub', 'EAP']
    data = {
        'leadtimes': leadtimes,
        'return_periods': return_periods,
        'POD': {model: {'observation': [], 'impact': []} for model in models},
        'FAR': {model: {'observation': [], 'impact': []} for model in models}
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
            
            for leadtime in leadtimes:
                for return_period in return_periods:
                    # Build the filename
                    file_name = f'scores_byCommuneRP{return_period}_yr_leadtime{leadtime}.csv'
                    file_path = os.path.join(comp_dir, file_name)
                    
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}, skipping.")
                        # Append NaN placeholders for missing data
                        data['POD'][model][comparison_type].append(None)
                        data['FAR'][model][comparison_type].append(None)
                        continue
                    
                    # Read the CSV file
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}, skipping.")
                        data['POD'][model][comparison_type].append(None)
                        data['FAR'][model][comparison_type].append(None)
                        continue
                    
                    # Filter rows for the specified admin unit
                    row = df[df['ADM2'] == admin_unit]
                    if row.empty:
                        print(f"Admin unit {admin_unit} not found in file: {file_path}, skipping.")
                        data['POD'][model][comparison_type].append(None)
                        data['FAR'][model][comparison_type].append(None)
                    else:
                        # Extract POD and FAR values
                        data['POD'][model][comparison_type].append(row['pod'].values[0] if 'pod' in row else None)
                        data['FAR'][model][comparison_type].append(row['far'].values[0] if 'far' in row else None)
    
    return data
