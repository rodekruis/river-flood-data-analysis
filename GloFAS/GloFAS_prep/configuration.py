from pathlib import Path 
import numpy as np
import math
import os
################################ GLOFAS ###############################################

os.chdir (f'C:/Users/els-2/') 
cur = Path.cwd() 
DataDir = cur / 'MaliGloFAS/data'
# Mali area coordinates as in notation as used by GloFAS (which is very weird)
# north, west, south, east
MaliArea = [25, -12.25, 10, 4.25] # Mali area [lat_max, lon_min, lat_min, lon_max] (upper left corner , down right corner)
regionPath = DataDir / f"Visualization/ADM1_Affected_SHP.shp" #région corresponds to ADM1, larger districts
communePath = DataDir / f"Visualization/ADM3_Affected_SHP.shp"
cerclePath = DataDir / f"Visualization/mli_admbnda_adm2_1m_gov_20211220.shp"
adminPaths = [regionPath, cerclePath, communePath]
lakesPath = DataDir / f'Visualization/waterbodies/waterbodies_merged.shp'
stationsDir = DataDir / f'stations'
DNHstations = stationsDir / f"Stations_DNH.csv"
googlestations = stationsDir / 'coords_google_gauges_Mali.csv'
GloFASstations = stationsDir / 'GloFAS_to_DNH_resembling_uparea.csv'
impact_csvPath = DataDir / "impact/MergedImpactData.csv"
settlements_tif = DataDir / "GlobalHumanSettlement/GHS_BUILT_S_E2030_GLOBE_R2023A_54009_100_V1_0.tif"

crs = f'EPSG:4326' 
RPsyr = [1.5, 2.0, 5.0, 10.0] # return period threshold in years
percentiles = [95, 98, 99] 
leadtimes = [72, 96, 120,144, 168] # add also 24hours
startYear = 2004 # could be 2004 but needs to be 2016 since there is no google data available before 
endYear = 2023 # 00:00 1st of january of that year, so up to but not including
triggerProb = 0.6
actionLifetime = 10 #days
adminLevel = 2 # choose level on which you would like to aggregate : 1,2,3
years = np.arange(startYear, endYear, 1)
admPath = adminPaths [(adminLevel-1)] # generate the useful administrative unit path 
nrCores = 4 #determine number of cpu cores to use (check your local device or the maximum allowed by your virtual computer)
measure = 'max' # measure to aggregate on :) 
# current EAP Propagation Trigger Model
StationCombos = [
    {"Upstream": "Banankoro", "Downstream": "Bamako", "PropagationTime": 4},
    {"Upstream": "Bamako", "Downstream": "Koulikoro", "PropagationTime": 1},
    {"Upstream": "Koulikoro", "Downstream": "Tamani", "PropagationTime": 3},
    {"Upstream": "Tamani", "Downstream": "Kirango", "PropagationTime": 2},
    {"Upstream": "Kirango", "Downstream": "Ké-Macina", "PropagationTime": 1},
]