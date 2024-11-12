def unzipGloFAS (DataDir, leadtime, month, day, year=None):
    '''month : month as number as string, so January = '01' (Type:Str)
        day : day as number as string, so first day of the month = '01' (Type:Str)
        year : is in file name if not forecast but reforecast'''
    if year==None:
        zipRasterPath = f'{DataDir}/GloFASreforecast/{int(leadtime)}hours/cems-glofas-reforecast_{month}_{day}.zip'
        with zipfile.ZipFile(zipRasterPath, 'r') as zip_ref:
            zip_ref.extractall(f'{DataDir}/GloFASreforecast/{int(leadtime)}hours/cems-glofas-reforecast_{month}_{day}')    
        rasterPath = f'{DataDir}/GloFASreforecast/{int(leadtime)}hours/cems-glofas-reforecast_{month}_{day}/data.grib'

    elif isinstance (year, (str,int)):
        year = str(year)
        zipRasterPath = f'{DataDir}/GloFASforecast/{int(self.leadtime)}hours/cems-glofas-forecast_{year}_{month}_{day}.zip'
        with zipfile.ZipFile(zipRasterPath, 'r') as zip_ref:
            zip_ref.extractall(f'{DataDir}/GloFASforecast/{int(self.leadtime)}hours/cems-glofas-forecast_{year}_{month}_{day}/')    
        rasterPath = f'{DataDir}/GloFASforecast/{int(self.leadtime)}hours/cems-glofas-forecast_{year}_{month}_{day}/'
    else: 
        ValueError (f"Argument year should be of type str or int")
    return rasterPath

def openGloFAS(rasterPath, lakesPath=None):
    '''Returns a DataArray with masked lakes for a single netcdf / grib file'''

    if rasterPath.endswith('grib'):
        Q_ds = xr.load_dataset(rasterPath, engine='cfgrib')
    elif rasterPath.endswith('nc'):
        Q_ds = xr.load_dataset(rasterPath)
    else: 
        raise TypeError('Make sure predictive GloFAS raster is either of format netCDF4 or GRIB2')

    # Correct longitude transformation if needed (see GloFAS issues)
    Q_ds["longitude"] = Q_ds["longitude"].where(Q_ds["longitude"] < 180, Q_ds["longitude"] - 360)

    Q_da = Q_ds["dis24"]
    Q_da.rio.write_crs(self.crs, inplace=True)

    if str(lakesPath).endswith('shp'):
        with fiona.open(lakesPath, 'r') as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        Q_da = Q_da.rio.clip(shapes, invert=True)

    return Q_da

