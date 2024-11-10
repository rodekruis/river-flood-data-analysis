# src/extract/export.py

from .call_ListGauges import get_ListGauges
from .call_GetGaugeModel import get_GetGaugeModel
from .call_QueryGaugeForecasts import get_QueryGaugeForecasts

import os
from typing import Tuple
import datetime
import pandas as pd


def export_data_to_csv(path: str, df: pd.DataFrame, idx = False) -> None:
    """
    Helper function to export a dataframe to a csv file

    :param path: path to the csv file
    :param df: dataframe to be exported
    :param idx: export index yes/no
    """
    df.to_csv(
        path,
        index = idx,
        decimal = '.',
        sep = ';',
        encoding = 'utf-8'
    )


def extract_country_data_for_time_delta(
        path_API_key: str,
        country: str,
        delta: Tuple[datetime.datetime, datetime.datetime],
        export: bool = True) -> pd.DataFrame:
    """
    Combines the calls of the
    - ListGauges
    - GetGaugeModel
    - QueryGaugeForecasts
    functions to extract the data for a given country and time delta.

    Data can be exported to a csv file optionally.
    All created dataframes are returned as a dictionary.

    :param path_API_key: path to the API key file
    :param country: country to extract data from
    :param delta: tuple of two datetime objects (forming a time delta)
    :param export: export data to csv file yes/no
    """
    print(f'Extracting data for {country} from {str(delta[0])[:10]} to {str(delta[1])[:10]}')

    df_gauges = get_ListGauges(country, path_API_key)
    if export:
        export_data_to_csv(
            f"../data/processed/ListGauges/{country}_gauges_listed.csv",
            df_gauges
        )
    df_gauge_models = get_GetGaugeModel(path_API_key, df_gauges)
    if export:
        export_data_to_csv(
            f"../data/processed/GetGaugeModel/{country}_gauge_models_metadata.csv",
            df_gauge_models
        )
    df_gauge_forecasts = get_QueryGaugeForecasts(
        path_API_key, 
        df_gauge_models['gaugeId'].tolist(), 
        delta
    )
    if export:
        os.makedirs(f"../data/floods_data/{country.lower()}", exist_ok = True)
        export_data_to_csv(
            f"../data/floods_data/{country.lower()}/{str(delta[0])[:10]}_to_{str(delta[1])[:10]}.csv",
            df_gauge_forecasts,
            True
        )

    return df_gauges, df_gauge_models, df_gauge_forecasts


def get_country_gauge_coords(df_gauges: pd.DataFrame) -> pd.DataFrame:
    """
    Return the DataFrame with gauge names and coordinates of a specific country

    :param df_gauges: DataFrame with gauge information
    :param country_name: Name of the country
    :return: DataFrame with gauge names and coordinates of a specific country
    """
    return df_gauges.set_index('gaugeId')[['latitude', 'longitude']]


def export_country_gauge_coords(
        df_gauges: pd.DataFrame, out: bool = False, country_name: str = None) -> None:
    """
    Export gauge names and coordinates of a specific country to .csv.
    Optionally prints them as well (default = False)

    :param df_gauges: DataFrame with gauge information
    :param country_name: Name of the country
    """
    df_subset = get_country_gauge_coords(df_gauges)
    df_subset.to_csv(f"../data/processed/gauge_coords/{country_name}_gauge_coords.csv",
                     index = True,
                     sep = ';',
                     decimal = '.',
                     encoding = 'utf-8')
    
    if out:
        print(f'Coordinates of gauges in {country_name}')
        print(df_subset)