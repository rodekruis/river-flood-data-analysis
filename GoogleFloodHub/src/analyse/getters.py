# src/analyse/getters.py

from .transform import convert_country_code_to_iso_a3

import datetime
import pandas as pd
import geopandas as gpd


def import_ListGauges_data(country: str) -> pd.DataFrame:
    """
    Imports the list of gauges for a given country from data/processed/ListGauges

    :param country: the country for which the list of gauges should be imported
    :return a dataframe containing the .csv data
    """
    return pd.read_csv(
        f"../data/processed/ListGauges/{country.lower()}_gauges_listed.csv",
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


def get_country_data(country: str, a: str, b: str) -> pd.DataFrame:
    """
    Imports three pieces of data for a given country:
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