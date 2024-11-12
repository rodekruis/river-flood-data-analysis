# 
from GloFAS.GloFAS_prep.aggregation import aggregation

def exceedance (threshold_gdf, nrEnsemble, measure='max'):
    '''GloFAS_gdf based on zonal stats using one ensemble member and timestamp: so only two dimensions '''
    GloFAS_gdf = aggregation (Q_da, threshold_gdf, 'polygon', nrEnsemble, measure=measure)
    exceedance = GloFAS_gdf[f'{measure}'] > self.threshold_gdf[f'{measure}']
    return exceedance

print (f'starting for {month, day}')
self.Q_da = self.openGloFAS(month, day)
print (f'for month, day {month, day}, glofas raster opened')
for timestamp in self.Q_da.time: # which will be the same month,day, but alternating years
    startLoop = datetime.now()
    valid_timestamp = pd.to_datetime(str(timestamp.values)) + timedelta(hours=self.leadtime)
    probability = probability


def probability(Q_da, threshold_gdf, nrCores, measure='max'):
    '''calculates probability for one month/day/year, so put in for loop where you loop over each grib/netcdf file, and then again for the date that is in that netcdf file'''

    nrMembers = len(Q_da.number)  # Number of ensemble members
    commune_info_df = threshold_gdf[[f'ADM{self.adminLevel}', 'geometry']].copy()
    floodedCommune_data = []
    exceedance_count = np.zeros(len(self.threshold_gdf), dtype=int)
        
    # Parallelize within the ensemble members
    results = Parallel(n_jobs=nrCores)(delayed(exceedance)(threshold_gdf, nrEnsemble, measure=measure) for ensemble in Q_da.number)
    
    for result in results:
        exceedance_count += result
        probability = exceedance_count / nrMembers

        flooded_df = pd.DataFrame({
            f'ADM{self.adminLevel}': self.thresholdCommuneMax_gdf[f'ADM{self.adminLevel}'],
            'ExceedanceCount': exceedance_count,
            'Probability': probability,
            'ValidTime': valid_timestamp
                })

        flooded_df = flooded_df.join(commune_info_df.set_index(f'ADM{self.adminLevel}'), on=f'ADM{self.adminLevel}')
        floodedCommune_data.append(flooded_df)

        print(f'timestep {valid_timestamp} done, took: {datetime.now() - startLoop}') 
    
    floodedCommune_gdf = pd.concat(floodedCommune_data, ignore_index=True)
    floodedCommune_gdf = gpd.GeoDataFrame(floodedCommune_gdf, geometry='geometry')
    return floodedCommune_gdf
