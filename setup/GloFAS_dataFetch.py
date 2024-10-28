from pathlib import Path
import cdsapi 

from datetime import datetime, timedelta
import os 

'''
Download data from glofas website, make sure you have a cds api token / key
, by following instructions here; https://ewds.climate.copernicus.eu/how-to-api#install-the-cds-api-token
1. First register at the cds store
2. Store a file at your homedirectory called .cdsapirc (so : $HOME/.cdsapirc)
3. The file is a text like :   
  url: https://ewds.climate.copernicus.eu/api
  key: <PERSONAL-ACCESS-TOKEN>  (visible on https://ewds.climate.copernicus.eu/how-to-api#install-the-cds-api-token once logged in)
BUT, on the original cds website they suggest using the old cds key: https://cds.climate.copernicus.eu/api - this is only useful for the ERA5 data etc, but we are interested in early warning :) so ewds 

'''
def get_monthsdays():
    start, end = datetime (2023,3,27), datetime(2024, 3, 25) #, datetime.today() #Reference dates for version 4 just 
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    monthday = []
    for d in days:
        if d.weekday() in [0, 3]:
            day_str = d.strftime("%m-%d")
            if day_str == "02-29":
                day_str = "02-28"  # Replace "02-29" with "02-28"
            monthday.append(day_str.split("-"))
    return monthday

def DataFetch (DataDir, leadtime, startYear, endYear, area):
    '''
    DataDir: path to datadirectory in which you want to download the data (type: Path or Str)
    leadtime: leadtime in hours as available by glofas: 24/48/72/120/144/168/etc  (type: Int)
    startYear: first year of interest (type: Int)
    endYear: # 00:00 1st of january of that year, so up to but not including this year in the download (type: Int)
    area: coordinates of the corner of the area of interest as a list: so [North, West, South, East] in decimal form: northern latitudes are positive, southern negative. eastern longitudes are positive, western longitudes negative
    '''
    MONTHSDAYS = get_monthsdays()
    c = cdsapi.Client()   
    DATASET='cems-glofas-reforecast'
    BBOX = area # [North, West, South, East] 
    YEARS  = ['%d'%(y) for y in range(startYear, endYear)]
    LEADTIMES = leadtime    # submit request
    for md in MONTHSDAYS:       
        month = md[0].lower()
        day = md[1]
        print (f'downloading for years: {startYear} - {endYear}, dd-mm: {day}-{month}')    
        REQUEST= {
                'system_version': ["version_4_0"],
                'variable': 'river_discharge_in_the_last_24_hours',
                'hydrological_model': 'lisflood',
                'product_type': 'ensemble_perturbed_reforecast',
                'area': BBOX,
                'hyear': YEARS,
                'hmonth': month,
                'hday': day,
                'leadtime_hour': LEADTIMES,
                'data_format': "grib2",
                'download_format': "zip"
                }
        DownloadDir= Path(f'{DataDir}/{int(leadtime)}hours/GloFAS/')
        DownloadDir.mkdir(parents=True, exist_ok=True)
        os.chdir (DownloadDir)
        client = cdsapi.Client()
        c.retrieve(DATASET, REQUEST).download(f'{DATASET}_{month}_{day}.zip')  


