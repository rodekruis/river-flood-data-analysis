from GloFAS.GloFAS_data_extractor.reforecast_dataFetch import get_monthsdays
from GloFAS.GloFAS_analysis.probability_exceeding_calculator import probability
def concatenate_prob():
    '''Concatenate GeoDataFrames over multiple day,months along the time dimension
        order may be odd, but timestamp provided is still right'''
    all_years_data = []
    MONTHSDAYS = get_monthsdays()
    for md in MONTHSDAYS:       
        month = md[0].lower()
        day = md[1]
        floodedCommune_gdf = probability()  # Open the GloFAS data for the current month, day !!!!!!!!!!!!! need to change now !
        all_years_data.append(floodedCommune_gdf)


    # Concatenate all GeoDataFrames for different years
    floodedCommune_gdf_concat = gpd.GeoDataFrame(pd.concat(all_years_data, ignore_index=True), geometry='geometry')
    floodedCommune_gdf_concat.to_file(f"{self.DataDir}/floodedRP{self.RPyr}yr_leadtime{self.leadtime}_ADM{self.adminLevel}.gpkg", driver="GPKG")

    return floodedCommune_gdf_concat