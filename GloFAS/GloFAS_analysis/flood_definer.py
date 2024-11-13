from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
floodProbability_path = self.DataDir/ f"floodedRP{self.RPyr}yr_leadtime{self.leadtime}_ADM{self.adminLevel}.gpkg"

def stampTrigger (floodProbability_gdf, actionLifetime, triggerProb):
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
