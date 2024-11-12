import cdsapi
import datetime
import warnings
import batch_configuration as cfg 
import os
from pathlib import Path

'''
Download forecast data from glofas website, make sure you have a cds api token / key
, by following instructions here; https://ewds.climate.copernicus.eu/how-to-api#install-the-cds-api-token
1. First register at the cds store
2. Store a file at your homedirectory called .cdsapirc (so : $HOME/.cdsapirc)
3. The file is a text like :   
  url: https://ewds.climate.copernicus.eu/api
  key: <PERSONAL-ACCESS-TOKEN>  (visible on https://ewds.climate.copernicus.eu/how-to-api#install-the-cds-api-token once logged in)
BUT, on the original cds website they suggest using the old cds key: https://cds.climate.copernicus.eu/api - this is only useful for the ERA5 data etc, but we are interested in early warning :) so ewds 

NB: difference in the approach between reforecast / forecast. see best practices page at ewds 
'''

def compute_dates_range(start_date,end_date,loop_days=True):
  
  
    start_date = datetime.date(*[int(x) for x in start_date.split('-')])
      
    end_date = datetime.date(*[int(x) for x in end_date.split('-')])
      
    ndays =  (end_date - start_date).days + 1
      
    dates = []
    for d in range(ndays):
        dates.append(start_date + datetime.timedelta(d))
      
    if not loop_days:
        dates = [i for i in dates if i.day == 1]
    else:
        pass
    return dates
  
  
  
if __name__ == '__main__':
    DataDir = cfg.DataDir
  
    # start the client
    c = cdsapi.Client()
  
  
    # user inputs
    DATASET='cems-glofas-forecast'
     
    START_DATE = '2024-07-26'
  
    END_DATE = '2024-11-01'
  
    LEADTIMES =  168
  
  
    # loop over dates and save to disk
  
    dates = compute_dates_range(START_DATE,END_DATE)
  
    for date in dates:
  
        year  = date.strftime('%Y')
        month = date.strftime('%m')
        day   = date.strftime('%d')
  
        print(f"RETRIEVING: {year}-{month}-{day}-{DATASET}")
        REQUEST={
                'system_version':'operational',
                'hydrological_model': 'lisflood',
                'product_type':'control_forecast',
                'variable': 'river_discharge_in_the_last_24_hours',
                'year': year,
                'month': month,
                'day': day,
                'leadtime_hour':LEADTIMES,
                'data_format': "grib2",
                'download_format': "zip"
                 }
        DownloadDir= Path(f'{DataDir}/GloFASforecast/{int(self.leadtime)}hours/')
        DownloadDir.mkdir(parents=True, exist_ok=True)
        os.chdir (DownloadDir)
        c.retrieve(DATASET, REQUEST).download(f'{DATASET}_{year}_{month}_{day}.zip')