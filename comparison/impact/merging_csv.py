import configuration as cfg 
import pandas as pd
from datetime import datetime, timedelta

DataDir = cfg.DataDir / 'impact'
MasterInnondationPath = DataDir / 'CleanedImpactInnondations_220623.csv'
DesinventarTextminingPath = DataDir / 'merged_desinventar_textmining.csv'

# Load the data
innondation_df = pd.read_csv(MasterInnondationPath)
desinventar_df = pd.read_csv(DesinventarTextminingPath)

import pandas as pd
from datetime import timedelta

# Convert dates to datetime objects, using dayfirst=True for day/month/year format
innondation_df['Start Date'] = pd.to_datetime(innondation_df['Start Date'], dayfirst=True)
innondation_df['End Date'] = pd.to_datetime(innondation_df['End Date'], dayfirst=True)
desinventar_df['Start Date'] = pd.to_datetime(desinventar_df['date'], dayfirst=True)
desinventar_df['End Date'] = pd.to_datetime(desinventar_df['date'], dayfirst=True)

# Normalize administrative units to uppercase in both datasets
innondation_df['Commune'] = innondation_df['Commune'].str.upper()
innondation_df['Cercle'] = innondation_df['Cercle'].str.upper()
innondation_df['Région'] = innondation_df['Région'].str.upper()

desinventar_df['Commune'] = desinventar_df['commune (adm3)'].str.upper()
desinventar_df['Cercle'] = desinventar_df['cercle (adm2)'].str.upper()
desinventar_df['Région'] = desinventar_df['region (adm1)'].str.upper()

# Function to split rows with multiple administrative units
def split_administrative_units(df):
    expanded_rows = []  # To store the expanded rows
    thrown_out_entries = []  # To store duplicates
    
    # Iterate through each row
    for idx, row in df.iterrows():
        # Split the values in 'Commune', 'Cercle', and 'Région' by comma (if they contain commas)
        communes = str(row['Commune']).split(',') if pd.notna(row['Commune']) else [None]
        cercles = str(row['Cercle']).split(',') if pd.notna(row['Cercle']) else [None]
        regions = str(row['Région']).split(',') if pd.notna(row['Région']) else [None]
        
        # For each combination of commune, cercle, and region, create a new entry
        for commune in communes:
            for cercle in cercles:
                for region in regions:
                    # Create a new row
                    new_row = row.copy()
                    new_row['Commune'] = commune.strip() if commune else None
                    new_row['Cercle'] = cercle.strip() if cercle else None
                    new_row['Région'] = region.strip() if region else None
                    
                    # Check if this entry is a duplicate by comparing it to existing expanded rows
                    is_duplicate = any(
                        (existing['Commune'] == new_row['Commune']) and
                        (existing['Cercle'] == new_row['Cercle']) and
                        (existing['Région'] == new_row['Région']) and
                        (existing['Start Date'] == new_row['Start Date']) and
                        (existing['End Date'] == new_row['End Date'])
                        for existing in expanded_rows
                    )
                    
                    # If it's not a duplicate, add it to the expanded rows
                    if not is_duplicate:
                        expanded_rows.append(new_row)
                    else:
                        # Log duplicate entry
                        admin_unit = new_row['Commune'] or new_row['Cercle'] or new_row['Région']
                        timestamp = f"{new_row['Start Date']} - {new_row['End Date']}"
                        print(f"Double entry for {admin_unit} at time {timestamp}")
    
    # Convert the list of expanded rows back into a DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df
# Limit events longer than 14 days to their first 10 days
def limit_long_events(df):
    df['Event Duration'] = (df['End Date'] - df['Start Date']).dt.days
    df.loc[df['Event Duration'] > 14, 'End Date'] = df['Start Date'] + timedelta(days=14)
    return df

# Convert dates to datetime objects if they aren't already
innondation_df['Start Date'] = pd.to_datetime(innondation_df['Start Date'], dayfirst=True)
innondation_df['End Date'] = pd.to_datetime(innondation_df['End Date'], dayfirst=True)

innondation_df = limit_long_events (innondation_df)
# Apply the splitting function
innondation_df_expanded = split_administrative_units(innondation_df)

# Function to check if two date ranges overlap
def is_date_overlap(start1, end1, start2, end2):
    return max(start1, start2) <= min(end1, end2)

# Function to check if two rows match based on administrative units
def is_admin_unit_match(row1, row2):
    # 1. First check if both communes are available and match
    if pd.notna(row1['Commune']) and pd.notna(row2['Commune']):
        return row1['Commune'] == row2['Commune']  # Match only if Communes are the same
    
    # 2. If Commune is missing in either dataset, check Cercle
    if pd.isna(row1['Commune']) or pd.isna(row2['Commune']):
        if pd.notna(row1['Cercle']) and pd.notna(row2['Cercle']):
            return row1['Cercle'] == row2['Cercle']  # Match only if Cercles are the same
    
    # 3. If Commune and Cercle are missing, check Région
    if (pd.isna(row1['Commune']) and pd.isna(row2['Commune'])) or (pd.isna(row1['Cercle']) and pd.isna(row2['Cercle'])):
        return row1['Région'] == row2['Région']  # Match only if Régions are the same
    
    return False  # No match if none of the above conditions apply

# Merge and deduplicate while adding additional info where relevant
def merge_with_additional_info(innondation_df, desinventar_df):
    merged_data = innondation_df.copy()
    thrown_out_entries = []

    # Iterate over desinventar entries
    for _, row2 in desinventar_df.iterrows():
        overlap_found = False
        
        # Iterate over innondation entries
        for idx1, row1 in merged_data.iterrows():
            same_admin_unit = is_admin_unit_match(row1, row2)  # Check for matching administrative unit
            
            # If same administrative unit and date overlap
            if same_admin_unit and is_date_overlap(row1['Start Date'], row1['End Date'], row2['Start Date'], row2['End Date']):
                overlap_found = True
                
                # Compare date ranges, update if desinventar has more information
                duration1 = (row1['End Date'] - row1['Start Date']).days
                duration2 = (row2['End Date'] - row2['Start Date']).days
                if duration2 > duration1:
                    # Update innondation row with longer coverage or missing data from desinventar
                    merged_data.at[idx1, 'Start Date'] = row2['Start Date']
                    merged_data.at[idx1, 'End Date'] = row2['End Date']
                
                # Add extra administrative info if desinventar has more detail
                if pd.isna(row1['Commune']) and pd.notna(row2['Commune']):
                    merged_data.at[idx1, 'Commune'] = row2['Commune']
                if pd.isna(row1['Cercle']) and pd.notna(row2['Cercle']):
                    merged_data.at[idx1, 'Cercle'] = row2['Cercle']
                if pd.isna(row1['Région']) and pd.notna(row2['Région']):
                    merged_data.at[idx1, 'Région'] = row2['Région']
                
                break  # Stop searching once overlap is found
        
        # If no overlap found, append the desinventar row
        if not overlap_found:
            merged_data = pd.concat([merged_data, pd.DataFrame([row2])], ignore_index=True)
    
    return merged_data

# Merge datasets
merged_df = merge_with_additional_info(innondation_df_expanded, desinventar_df)

# Save final merged data
merged_df.to_csv(DataDir / 'MergedImpactData.csv', index=False)
