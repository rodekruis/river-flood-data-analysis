from pathlib import Path 
import numpy as np
import math
import os

os.chdir (f'C:\\Users\\els-2\\')
cur = Path.cwd() 
DataDir = cur / 'MaliGloFAS\\data'
# Mali area coordinates as in notation as used by GloFAS (which is very weird)
# north, west, south, east
MaliArea = [25, -12.25, 10, 4.25] # Mali area [lat_max, lon_min, lat_min, lon_max] (upper left corner , down right corner)
regionPath = DataDir / f"Visualization/ADM1_Affected_SHP.shp" #région corresponds to ADM1, larger districts
communePath = DataDir / f"Visualization/ADM3_Affected_SHP.shp"
cerclePath = DataDir / f"Visualization/mli_admbnda_adm2_1m_gov_20211220.shp"
adminPaths = [regionPath, cerclePath, communePath]
lakesPath = DataDir / f'Visualization/waterbodies/waterbodies_merged.shp'
stationsDir = DataDir / f'stations'

googlestations = stationsDir / 'coords_google_gauges_Mali.csv'
GloFASstations = stationsDir / 'GloFAS_MaliStations_v4.csv'
impact_csvPath = DataDir / "impact/MergedImpactData.csv"
settlements_

crs = f'EPSG:4326' 
RPsyr = [1.5, 2.0, 5.0, 10.0] # return period threshold in years 
leadtimes = 168 # hours
startYear = 2004 
endYear = 2023 # 00:00 1st of january of that year, so up to but not including
triggerProb = 0.6
actionLifetime = 10 #days
adminLevel = 2 # choose level on which you would like to aggregate : 1,2,3
years = np.arange(startYear, endYear, 1)
admPath = adminPaths [(adminLevel-1)] # generate the useful administrative unit path 
nrCores = 6 #determine number of cpu cores to use
measure = 'max' # measure to aggregate on 