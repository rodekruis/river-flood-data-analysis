# src/analyse/getters.py

from .transform import convert_country_code_to_iso_a3

from typing import Dict, List, Tuple
import datetime
import pandas as pd
import geopandas as gpd
import xarray as xr


def import_ListGauges_data(country: str, unverified = False) -> pd.DataFrame:
    """
    Imports the list of gauges for a given country from data/processed/ListGauges

    :param country: the country for which the list of gauges should be imported
    :return a dataframe containing the .csv data
    """
    unverified_param = '_unverified' if unverified else ''
    return pd.read_csv(
        f"../data/processed/ListGauges/{country.lower()}_gauges_listed{unverified_param}.csv",
        sep = ';',
        decimal = '.',
        encoding = 'utf-8'
    )


def import_GetGaugeModel_data(country: str) -> pd.DataFrame:
    """
    Imports the gauge model data for a given country from data/processed/GetGaugeModel

    :param country: the country for which the gauge model data should be imported
    :return a dataframe containing the .csv data
    """
    return pd.read_csv(
        f"../data/processed/GetGaugeModel/{country.lower()}_gauge_models_metadata.csv",
        sep = ';',
        decimal = '.',
        encoding = 'utf-8'
    )


def validate_date_string(date: str) -> bool:
    """
    Validates a date string in the format YYYY-MM-DD

    :param date: the date string to validate
    :return True if the date string is valid, False otherwise
    """
    # Check if the date string is in the correct format and otherwise print what the correct format should be and return false
    try:
        datetime.datetime.strptime(date, '%Y-%m-%d')
        return True
    except ValueError:
        print("Incorrect data format, should be YYYY-MM-DD")
        return False


def import_country_forecast_data(country: str, a: str, b: str) -> pd.DataFrame:
    """
    Imports the forecast data for a given country from data/processed/GetGaugeModel.
    It needs as parameters the country and time delta (= starting and ending issue
    time) of interest such that the exact correct file can be imported

    :param country: the country for which the forecast data should be imported
    :param a: the starting issue time of interest
    :param b: the ending issue time of interest
    :return a dataframe containing the .csv data
    """
    if not validate_date_string(a) or not validate_date_string(b):
        return None
    return pd.read_csv(
        f"../data/floods_data/{country.lower()}/{a}_to_{b}.csv",
        index_col = 0,
        sep = ';',
        decimal = '.',
        encoding = 'utf-8'
    )


def get_country_data(country: str, a: str, b: str, unverified = False) -> pd.DataFrame:
    """
    Imports three pieces of data for a given country:
    - metadata per country (eg containing the gauges and their coordinates)
    - metadata per gauge (eg containing the danger levels)
    - forecast data per gauge

    When unverified is set to True, the gauges of unverified quality are
    also imported

    :param country: the country for which the data should be imported
    :param a: the starting issue time of interest
    :param b: the ending issue time of interest
    :param unverified: whether to import unverified gauges
    :return dataframes containing the three pieces of data
    """
    df_gauges = import_ListGauges_data(country, unverified)
    df_gauge_meta = import_GetGaugeModel_data(country)
    df_forecasts = import_country_forecast_data(country, a, b)

    return df_gauges, df_gauge_meta, df_forecasts


def get_country_data_unverified(country: str, a: str, b: str) -> pd.DataFrame:
    """
    Unverified version of get_country_data(), which means it also
    imports the gauges of ListGauges of unverified quality. ---
    Imports of three pieces of data for a given country:
    - metadata per country (eg containing the gauges and their coordinates)
    - metadata per gauge (eg containing the danger levels)
    - forecast data per gauge

    :param country: the country for which the data should be imported
    :param a: the starting issue time of interest
    :param b: the ending issue time of interest
    :return dataframes containing the three pieces of data
    """
    df_gauges = import_ListGauges_data(country)
    df_gauge_meta = import_GetGaugeModel_data(country)
    df_forecasts = import_country_forecast_data(country, a, b)

    return df_gauges, df_gauge_meta, df_forecasts


def get_shape_file(file : str) -> gpd.GeoDataFrame:
    """
    Get the shape file for a country

    :param country: the country
    :return: the GeoDataFrame
    """
    try:
        return gpd.read_file(f"../data/shape_files/{file}")
    except Exception as exc:
        raise Exception(f'Error reading shapefile: {exc}')
    

def get_country_polygon(country_code : str) -> gpd.GeoDataFrame:
    """
    Get the polygon of a country as a GeoDataFrame

    :param country_code: the country code
    :return: the polygon
    """
    gdf = get_shape_file('ne_110m_admin_0_countries')
    iso_a3 = convert_country_code_to_iso_a3(country_code)

    country_row = gdf[gdf['SOV_A3'] == iso_a3]
    if country_row.empty:
        raise ValueError(f'Country with ISO A3 code {country_code} not found')
    
    return gpd.GeoDataFrame(geometry = [country_row['geometry'].values[0]])


def get_severity_levels(df: pd.DataFrame, hybas: str) -> dict:
    """
    Get the severity levels for a hybas from the dataframe,
    where 'warningLevel' corresponds to a two-year return period,
    'dangerLevel' to a 5-year return period, and
    'extremeDangerLevel' to a 20-year return period.

    :param df: the dataframe
    :param hybas: the hybas
    :return: the severity levels
    """
    two_year = df[df['gaugeId'] == hybas]['warningLevel'].values[0]
    five_year = df[df['gaugeId'] == hybas]['dangerLevel'].values[0]
    twenty_year = df[df['gaugeId'] == hybas]['extremeDangerLevel'].values[0]
    return {
        'two_year': two_year,
        'five_year': five_year,
        'twenty_year': twenty_year
    }


def get_country_gauge_coords(country: str) -> pd.DataFrame:
    """
    Get the coordinates of the gauges in a country, stored in data/ folder

    :param country: name of the country
    :return: DataFrame with the gaugeId, latitude and longitude
    """
    if country[0].islower():
        country = country.capitalize()
    return pd.read_csv(f'../data/processed/gauge_coords/{country}_gauge_coords.csv',
                       index_col = None, sep = ';', decimal = '.')


def get_datasets_unit_with_most_gauges(
        dict_ds_agg: Dict[str, xr.Dataset],
        dict_ds: Dict[str, xr.Dataset]
    ) -> Tuple[List[xr.Dataset], str]:
    """
    Finds the administrative unit with the most gauges and returns 
    the gauge datasets belonging to that unit in a list, taken from
    the non-aggregated dictionary of datasets

    :param dict_ds_agg: dict with the datasets
    :param dict_ds: dict with the datasets
    :return: list with the datasets of the admin unit with most gauges and unit name
    """
    most_common_admin_unit = \
        max(dict_ds_agg.keys(), key = lambda k: len(dict_ds_agg[k].attrs['gauge_ids']))
    
    datasets_for_admin_unit = [
        ds for ds in dict_ds.values() if most_common_admin_unit in ds.attrs['admin_unit']
    ]

    return datasets_for_admin_unit, most_common_admin_unit


def get_datasets_for_xth_admin_unit(
        dict_ds_agg: Dict[str, xr.Dataset],
        dict_ds: Dict[str, xr.Dataset],
        x: int
    ) -> Tuple[List[xr.Dataset], str]:
    """
    Finds the administrative unit with the xth most gauges and returns 
    the gauge datasets belonging to that unit in a list, taken from
    the non-aggregated dictionary of datasets

    :param dict_ds_agg: dict with the datasets
    :param dict_ds: dict with the datasets
    :param x: the xth admin unit to get the datasets for
    :return: list with the datasets of the admin unit with most gauges and unit name
    """
    aus_sorted = sorted(dict_ds_agg.keys(),
                        key = lambda k: len(dict_ds_agg[k].attrs['gauge_ids']), 
                        reverse = True)
    if x < 1 or x > len(aus_sorted):
        raise ValueError(f'x must be between 1 and {len(aus_sorted)}')
    
    xth_admin_unit = aus_sorted[x - 1]
    datasets_for_admin_unit = [
        ds for ds in dict_ds.values() if xth_admin_unit in ds.attrs['admin_unit']
    ]

    # for the datasets in datasets_for_admin_unit, see if the
    # gauge ID's are correct by comparing them to the gauge ID
    # list in the attributes of the dataset for the admin unit
    gauge_ids = dict_ds_agg[xth_admin_unit].attrs['gauge_ids']
    for ds in datasets_for_admin_unit:
        if ds.attrs['gauge_id'] not in gauge_ids:
            raise ValueError(f'gauge ID {ds.attrs["gauge_id"]} not in gauge ID list')

    return datasets_for_admin_unit, xth_admin_unit