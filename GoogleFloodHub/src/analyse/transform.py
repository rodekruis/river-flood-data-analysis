# src/analyse/transform.py

from extract import get_json_file

import os
import datetime
import pandas as pd
import geopandas as gpd


def convert_df_to_gdf(df : pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert a DataFrame to a GeoDataFrame by taking the latitude and longitude columns

    :param df: the DataFrame
    :return: the GeoDataFrame
    """
    return gpd.GeoDataFrame(
        df,
        geometry = gpd.points_from_xy(df['longitude'], df['latitude']),
        crs = 'EPSG:4326' # Uniform projection to WGS84
    )


def convert_country_code_to_iso_a3(country_code : str) -> str:
    """
    Convert a country code to an ISO A3 code

    :param country_code: the country code
    :return: the ISO A3 code
    """
    return get_json_file(
        "../data/mappings/country_codes_to_ISO_A3.json"
        )[country_code]


def subset_country_gauge_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts coordinates from dataframe generated by GetGaugeModel query

    :param df: dataframe containing gauge metadata
    :return: dataframe with gauge coordinates
    """
    return df[['gaugeId', 'latitude', 'longitude']]


def make_subset_for_gauge_and_issue_time(
        df: pd.DataFrame, gauge: str, issue_date: datetime.datetime) -> pd.DataFrame:
    """
    Create a subset of a DataFrame for a specific gauge and issue time

    :param df: DataFrame with forecasted values
    :param gauge: ID of the gauge
    :param issue_date: issue time of the forecast
    :return: subset of the DataFrame
    """
    if not all(df['gaugeId'].apply(lambda x: isinstance(x, str))):
        df['gaugeId'] = df['gaugeId'].astype(str)
    if not all(df['issue_date'].apply(lambda x: isinstance(x, datetime.datetime))):
        df['issue_date'] = pd.to_datetime(df['issue_date'])
    assert all(df['gaugeId'].apply(lambda x: isinstance(x, str))), \
        "gaugeId column contains non-string values, subseting hindered"
    assert all(df['issue_date'].apply(lambda x: isinstance(x, datetime.datetime))), \
        "issue_date column contains non-date values, subsetting hindered"
    
    # For plotting purposes, the time can (and must) be normalized
    df['issue_date'] = df['issue_date'].dt.normalize()

    return df[(df['gaugeId'] == gauge) & (df['issue_date'] == issue_date)]


def add_shape_data_to_df(df: pd.DataFrame, path: str) -> gpd.GeoDataFrame:
    """
    Adds shape data to a DataFrame with admin units
    """
    gpd_adm_units = gpd.read_file(path)
    gpd_adm_units = gpd_adm_units[['ADM2_PCODE', 'ADM2_FR', 'geometry']]
    df.rename(columns = {'identifier': 'admin_unit'}, inplace = True)
    gpd_adm_units.rename(columns = {'ADM2_PCODE': 'admin_unit'}, inplace = True)
    merged_df = pd.merge(df, gpd_adm_units, on = 'admin_unit', how = 'left')
    merged_gdf = gpd.GeoDataFrame(merged_df, geometry = 'geometry', crs = gpd_adm_units.crs)
    return merged_gdf


def export_all_results_to_geojson(results_folder: str, geojson_folder: str) -> None:
    """
    Loop through results, merge with shapefile data, export to geojson
    """
    path_shp = '../data/shape_files/mali_ALL/mli_adm_ab_shp/mli_admbnda_adm2_1m_gov_20211220.shp'

    if not os.path.exists(geojson_folder):
        os.makedirs(geojson_folder)

    for file in os.listdir(results_folder):
        if file.startswith('GFH_vs_IMPACT'):
            csv_path = os.path.join(results_folder, file)
            df = pd.read_csv(csv_path, index_col = None)
            gdf = add_shape_data_to_df(df, path_shp)
            out_geojson_path = os.path.join(geojson_folder, file.replace('.csv', '.geojson'))
            gdf.to_file(out_geojson_path, driver = 'GeoJSON')