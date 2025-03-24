# src/analyse/tests.py

from typing import Any, Dict, List
import pandas as pd
import geopandas as gpd
import xarray as xr

# Contains some test(s) that check whether some data transformations have
# been done correctly. But, beware: the tests are not exhaustive in any way


def assert_same_coord_system(gpd_1: gpd.GeoDataFrame, gpd_2: gpd.GeoDataFrame) -> bool:
    """
    Small check whether the coordinate systems of two
    GeoDataFrames are the same. Can be used in another test

    :param gpd_1: first GeoDataFrame
    :param gpd_2: second GeoDataFrame
    """
    assert gpd_1.crs == gpd_2.crs, "Coordinate systems are not the same"


def assure_all_thresholds_added(
        dict_rf: Dict[str, Any],
        common_keys: List[str],
        rps: List[str],
        percentiles: List[int]
    ) -> None:
    """
    Loops over all datasets and checks if all thresholds were added

    :param dict_rf: dictionary with reforecast datasets
    :param common_keys: the common keys between the dictionaries
    :param rps: the return periods to calculate
    :param percentiles: the percentiles to calculate
    """
    for key in common_keys:
        ds_rf = dict_rf[f'ds_{key}']
        for rp in rps:
            if f'RP_{rp}' not in ds_rf.attrs:
                raise ValueError(f'RP_{rp} not added to {key}')
        for perc in percentiles:
            if f'pc_{perc}th' not in ds_rf.attrs:
                raise ValueError(f'pc_{perc}th not added to {key}')
            


def assure_admin_units_assigned(dict_ds: Dict[str, xr.Dataset]) -> None:
    """
    Check if all datasets have been assigned an admin unit

    :param dict_ds: dict of datasets
    """
    for ds in dict_ds.values():
        if 'admin_unit' not in ds.attrs:
            print(f'No admin unit assigned to dataset {ds.attrs["gauge_id"]}')
            continue
        admin_unit = ds.attrs['admin_unit']
        if any(pd.isna(unit) for unit in admin_unit):
            print(f'No admin unit assigned to dataset {ds.attrs["gauge_id"]} (NaN found)')


def assure_attributes_assigned(dict_datasets: Dict[str, xr.Dataset]):
    """ 
    Check if all datasets have been assigned an admin unit,
    including coordinates, gauge_id, and qualityVerified

    :param dict_datasets: dict of datasets
    """
    for ds in dict_datasets.values():
        assert 'gauge_id' in ds.attrs, f'No gauge ID assigned to dataset {ds}'
        assert 'qualityVerified' in ds.attrs, f'No qualityVerified assigned to dataset {ds.attrs["gauge_id"]}'
        assert 'admin_unit' in ds.attrs, f'No admin unit assigned to dataset {ds.attrs["gauge_id"]}'
        assert 'latitude' in ds.attrs, f'No latitude assigned to dataset {ds.attrs["gauge_id"]}'
        assert 'longitude' in ds.attrs, f'No longitude assigned to dataset {ds.attrs["gauge_id"]}'