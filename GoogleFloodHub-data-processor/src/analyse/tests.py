# src/analyse/tests.py

# Contains some test(s) that check whether some data transformations have
# been done correctly. But, beware: the tests are not exhaustive in any way

import pandas as pd
import geopandas as gpd


def assert_same_coord_system(gpd_1: gpd.GeoDataFrame, gpd_2: gpd.GeoDataFrame) -> bool:
    """
    Small check whether the coordinate systems of two
    GeoDataFrames are the same. Can be used in another test

    :param gpd_1: first GeoDataFrame
    :param gpd_2: second GeoDataFrame
    """
    assert gpd_1.crs == gpd_2.crs, "Coordinate systems are not the same"