# src/analyse/datasets.py

from .getters import get_country_gauge_coords
from .getters import get_shape_file
from .thresholds import add_thresholds_to_dataset
from .tests import assure_admin_units_assigned
from .tests import assert_same_coord_system

from typing import Any, Dict, List, Tuple
import os
import warnings
import pandas as pd
import geopandas as gpd
import xarray as xr


# Functions that make general modifications to the xarray GRRR datasets


def add_quality_verified_flag(
        d_ds: Dict[str, xr.Dataset],
        df_flags: pd.DataFrame
    ) -> xr.Dataset:
    """ Adds an attribute to each dataset in the dictionary of datasets
    from the column 'qualityVerified' in the DataFrame of flags,
    corresponding to the matching 'gaugeId' column in the df.

    Keys in the dict look like: 'ds_1120040380',
    and the gaugeId column like: 'hybas_1120040380',
    so the strings need to be adjusted to match.

    :param d_ds: dictionary of datasets
    :param df_flags: DataFrame with cols 'gaugeId' and 'qualityVerified'
    :return: dictionary of datasets with added attribute 'qualityVerified'
    """
    for key in d_ds.keys():
        # if key.startswith('ds_reforecast_'):
        #     gauge_id = key.replace('ds_reforecast_', 'hybas_')
        # elif key.startswith('ds_reanalysis_'):
        #     gauge_id = key.replace('ds_reanalysis_', 'hybas_')
        # else:
        #     raise ValueError(f'Unknown key: {key}')
        gauge_id = key.replace('ds_', 'hybas_')
        quality_verified = df_flags[df_flags['gaugeId'] == gauge_id]['qualityVerified'].values[0]
        d_ds[key].attrs['qualityVerified'] = quality_verified

    return d_ds


def index_country_gauge_coords(df_gauges: pd.DataFrame) -> pd.DataFrame:
    """
    Return the DataFrame with gauge names and coordinates of a specific country

    :param df_gauges: DataFrame with gauge information
    :param country_name: Name of the country
    :return: DataFrame with gauge names and coordinates of a specific country
    """
    return df_gauges.set_index('gaugeId')[['latitude', 'longitude']]


def export_country_gauge_coords(
        df_gauges: pd.DataFrame, country_name: str = None
    ) -> None:
    """
    Export gauge names and coordinates of a specific country to .csv.
    Optionally prints them as well (default = False)

    :param df_gauges: DataFrame with gauge information
    :param country_name: Name of the country
    """
    df_subset = index_country_gauge_coords(df_gauges)
    df_subset.to_csv(f"../data/processed/gauge_coords/{country_name}_gauge_coords.csv",
                     index = True,
                     sep = ';',
                     decimal = '.',
                     encoding = 'utf-8')


def assign_coords_to_datasets(
        datasets: Dict[str, xr.Dataset], country: str
    ) -> xr.Dataset:
    """
    Takes a dict of datasets and assigns their coordinates, which it gets
    from get_country_gauge_coords(), and assigns it to each dataset. The
    dict contains the names of the datasets as keys and the datasets as values

    :param datasets: dict of datasets
    :param country: name of the country
    :return: dict of datasets with coordinates
    """
    df_coords = get_country_gauge_coords(country)

    for gauge_id, dataset in datasets.items():
        # assumes full name, e.g. 'hybas_1120661040', in df_coords, thus creating
        # a comparison of solely the hybas numbers, not the full name or dataset identifier
        coords = df_coords[
            df_coords['gaugeId'].apply(lambda x: x.split('_')[-1]) == gauge_id.split('_')[-1]
        ]

        if not coords.empty:
            # add the coordinates to the dataset as attributes
            dataset.attrs['latitude'] = coords['latitude'].values[0]
            dataset.attrs['longitude'] = coords['longitude'].values[0]
            # add the hybas_id to the dataset as well (e.g. '1120661040')
            dataset.attrs['gauge_id'] = gauge_id.split('_')[-1]
        else:
            print(f'No coordinates found for gauge {gauge_id}') 

    return datasets


def create_coords_df_from_ds(dict_ds: Dict[str, xr.Dataset]) -> pd.DataFrame:
    """
    Create a DataFrame with all gauge ID's and coordinates in a
    dictionary with xarray Datasets

    :param dict_ds: xarray Dataset
    :return: DataFrame with the coordinates
    """
    return pd.DataFrame([
        {
            'gauge_id': ds.attrs['gauge_id'],
            'longitude': ds.attrs['longitude'],
            'latitude': ds.attrs['latitude']
        }
        for ds in dict_ds.values()
    ])


def handle_NaN_admin_units(codes: pd.Series) -> list:
    """
    Handle NaN values in the series of administrative unit assignments for gauges

    :param codes: Series containing administrative unit codes (ADM2_PCODE) for a gauge
    :return: list of valid administrative unit codes (ADM2_PCODE)
    """
    if codes.isna().any():
        print(f"Warning: Found NaN in administrative unit assignment "
              f"for gauge IDs: {codes[codes.isna()].index.tolist()}")
    return list(codes.dropna())


def assign_admin_unit_to_datasets(
        dict_ds: Dict[str, xr.Dataset],
        country: str = 'Mali',
        verbose: bool = False,
        # Path to the shape file with admin level 2 units for Mali
        path: str = 'mali_ALL/mli_adm_ab_shp/mli_admbnda_adm2_1m_gov_20211220.shp',
        buffer_radius: int = 5000
    ) -> Dict[str, xr.Dataset]:
    """
    Assigns the administrative unit to each dataset in the dictionary by:
    (1) assiging coordinates to the datasets, with assign_coords_to_datasets(),
        which takes information queried by ListGauges() in the 'extract' package
    (2) creating a GeoDataFrame from the dataset coordinates, which includes a
        5 km buffer around the gauges, to account for shape file inaccuracies and,
        more importantly, the fact that gauges are usually located in rivers, which,
        in turn, are usually borders between administrative units, causing gauges to
        be located in only onr of the units, while they effectively tell about both.
        With a buffer, this is accounted for, and, as a result, gauges can be assigned
        to multiple administrative units, if they simply intersect with multiple units
    (3) creating a GeoDataFrame from the shape file with the admin units, source:
        (https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/)
    (4) classifying the gauges into the administrative units by joining the two above
    (5) adding the admin unit names to the datasets by matching the gauge ID's and
        the found administrative units
    (6) returning the updated dictionary with the datasets, with datasets that now have
        the attributes 'longitude', 'latitude', and 'admin_unit'

    :param dict_ds: dict of datasets
    :param country: name of the country
    :param verbose: whether to print some test print-s's
    :param path: path to the shape file with the admin units
    :param buffer_radius: radius of the buffer around the gauges, standard is 5 km
    :return: dict of datasets with administrative units
    """
    #* (1): assign coordinates to the datasets
    dict_ds = assign_coords_to_datasets(dict_ds, country)
    # print(next(iter(dict_ds.items()))) if verbose else None

    #* (2): create a GeoDataFrame from the dataset coordinates;
    # geometry is a point for each gauge, with coords (x, y)
    df_gauge_coords = create_coords_df_from_ds(dict_ds)
    gpd_Mali_gauge_coords = gpd.GeoDataFrame(
        df_gauge_coords,
        geometry = gpd.points_from_xy(
            df_gauge_coords['longitude'], df_gauge_coords['latitude']
        ),
        crs = 'EPSG:4326'
    )
    # add a buffer of 5 km around the points to account for inaccuracies,
    # where 1 degree is approx. 111,32 km at the equator, so in degrees (which
    # we have to use as the coordinate system is WGS84), 5 km is 5000 meter \
    # divided by 111.320 meters (5000 / 111320). This is a rough estimate, but
    # should be sufficient, since the number of 5 km is too mostly arbirtrary.
    # Also, surpress the warning that the buffer is not exact, as we are aware
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpd_Mali_gauge_coords['geometry'] = \
            gpd_Mali_gauge_coords.geometry.buffer(buffer_radius / 111320)

    #* (3): read the shape file into a GeoDataFrame and convert it to WGS84
    # (which is the coordinate system used by the gauge data)
    gpd_adm_units_Mali = get_shape_file(path).to_crs('EPSG:4326')
    # check if the coord systems are the same
    if gpd_adm_units_Mali.crs != gpd_Mali_gauge_coords.crs:
        gpd_adm_units_Mali = gpd_adm_units_Mali.to_crs(gpd_Mali_gauge_coords.crs)
    assert_same_coord_system(gpd_adm_units_Mali, gpd_Mali_gauge_coords)

    #* (4) now we can classify the gauges into the administrative units:
    # creating a joined dataframe with the gauges as basis, meaning
    # that gauges get assigned to the admin unit they are within,
    # including their metadata (such as the shape of the admin unit).
    # (And, thus, the rest of the admin units are not considered.)
    gpd_gauges_classified = gpd.sjoin(
        gpd_Mali_gauge_coords, gpd_adm_units_Mali,
        how = 'left',           # joins left, i.e. the gauges serve as basis
                                # checks if the gauge intersects with the admin unit
        predicate = 'intersects',
        lsuffix = 'gauge', rsuffix = 'adm'
    )
    print(gpd_gauges_classified.head(1)) if verbose else None
    # make a mapping of the gauge ID's and the admin unit names:
    # group by gauge ID; select the admin unit names; check for NaNs;
    # convert to list; then dictionary with {gauge_id: [admin_unit]}
    mapping = gpd_gauges_classified.groupby('gauge_id')['ADM2_PCODE']\
        .apply(handle_NaN_admin_units).to_dict()

    #* (5) lastly, we add the admin unit names to the datasets
    print(mapping) if verbose else None
    for gauge_id, admin_units in mapping.items():
        dict_ds[f'ds_{gauge_id}'].attrs['admin_unit'] = admin_units

    #* (6) check result and return
    assure_admin_units_assigned(dict_ds)
    print('\n\n', next(iter(dict_ds.items()))) if verbose else None
    return dict_ds


def export_datasets_to_netcdf(
        d_ds: Dict[str, xr.Dataset], path: str, lt: int
    ) -> None:
    """
    Export a dataset to a netCDF file

    :param d_ds: dict of datasets
    :param path: path to the file
    :param lt: lead time of the dataset
    """
    for key, ds in d_ds.items():
        try:
            if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok = True)
            ds.to_netcdf(f'{path}{key}_{lt * 24}lt.nc')
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {fnf_error}")
        except PermissionError as perm_error:
            print(f"Permission error: {perm_error}")
        except TypeError as type_error:
            print(f"Type error: {type_error}")
        except Exception as exc:
            print(f"Failed to dataset to '{path}': {exc}")


def import_datasets_from_netcdf(path: str, lt: int) -> Dict[str, xr.Dataset]:
    """
    Import a dictionary of datasets from a folder with netCDF files:
    the inverse of export_datasets_to_netcdf()

    :param path: path to the files
    :param lt: lead time of the datasets to import
    :return: dict of datasets
    """
    d_ds = {}
    if not os.path.exists(path):
        raise ValueError('Path does not exist')
    for filename in os.listdir(path):
        if filename.endswith(f'_{lt * 24}lt.nc'):
            key = filename.split('_')[0]
            file_path = os.path.join(path, filename)
            try:
                ds = xr.open_dataset(file_path)
                d_ds[key] = ds
            except Exception as exc:
                print(f"Failed to import dataset from '{file_path}': {exc}")
    return d_ds


def create_admin_unit_set(dict_ds: Dict[str, xr.Dataset]) -> set:
    """
    Create a unique set of the admin units in the datasets

    :param dict_ds: dict of datasets
    :return: set of admin units
    """
    admin_units = set()
    for ds in dict_ds.values():
        if 'admin_unit' in ds.attrs:
            if ds.attrs['admin_unit'] is None:
                print(f'No admin unit found in dataset {ds.attrs["gauge_id"]}')
            else:
                admin_units.update(ds.attrs['admin_unit'])
        else:
            raise ValueError('No admin unit found in dataset')
    return admin_units


def get_dict_ds_per_admin_unit(
        dict_ds: Dict[str, xr.Dataset]
    ) -> Dict[str, list]:
    """ Get a dictionary with the datasets per admin unit

    :param dict_ds: dict of datasets
    :return: dict of datasets per admin unit
    """
    admin_units = create_admin_unit_set(dict_ds)
    dict_ds_per_admin_unit = {unit: [] for unit in admin_units}
    for ds in dict_ds.values():
        for unit in ds.attrs['admin_unit']:
            dict_ds_per_admin_unit[unit].append(ds)
    return dict_ds_per_admin_unit


def pretty_print_list(l: list) -> None:
    """
    Pretty print a list

    :param l: list
    """
    print(', '.join(l))


def print_gauge_ids_per_admin_unit(
        dict_ds: Dict[str, xr.Dataset]
    ) -> None:
    """ Prints the gauge IDs in an admin unit/district """
    print("[admin unit ID] : list([gauge ID's])")
    for unit, datasets in dict_ds.items():
        print(unit, end = ' : ')
        pretty_print_list([ds.attrs['gauge_id'] for ds in datasets])
    print('\n')


def subset_lead_time(ds: xr.Dataset, lt: int) -> xr.Dataset:
    """
    Subset the dataset to a certain lead time

    :param ds: xarray Dataset
    :param lt: lead time to subset to
    :return: xarray Dataset with subsetted lead time
    """
    if lt < 0 or lt > 7:
        raise ValueError('Lead time must be between 0 and 7 days')
    return ds.sel(lead_time = pd.Timedelta(days = lt))


def assign_actual_dates_to_rf_dataset(ds: xr.Dataset) -> xr.Dataset:
    """ Assign the actual dates to a reforecast dataset as new coordinates

    :param ds: xarray Dataset
    :return: xarray Dataset with actual dates as coordinates
    """
    actual_dates = ds['issue_time'] + ds['lead_time']
    return ds.assign_coords(actual_date = ('issue_time', actual_dates.data))


def assign_actual_dates_to_ra_dataset(ds: xr.Dataset) -> xr.Dataset:
    """ Assign the actual dates to a reanalysis dataset as new coordinates

    :param ds: xarray Dataset
    :return: xarray Dataset with actual dates as coordinates
    """
    actual_dates = ds['time']
    return ds.assign_coords(actual_date = ('time', actual_dates.data))


def aggregate_district_datasets_maximally(
        datasets: List[xr.Dataset],
        lt: int,
        district_name: str
) -> xr.Dataset:
    """ Takes in a dictionary of datasets belonging to one district,
    subsets lead time and converts to 'actual time', aggregates them
    by taking the maximum at every timestep, and returns a dataset
    with the aggregated timeseries and updates the attributes to:
        - identifier: district_name;
        - gauge_ids: list of gauge ID's in the district.
        
    :param datasets: list of datasets
    :param lt: lead time to subset to
    :param district_name: name of the district
    :return: aggregated dataset
    """
    # (1) concatenate the datasets into one dataset and add gauge_id dimension;
    ds_combined = xr.concat(datasets, dim = 'gauge_id')
    # (2) check if it contains the 'lead_time' dimension, if so, subset it and
    #     assign 'actual data' with the lead time, else, it must be a reanalysis
    #     dataset, for which we just assign the actual dates immediately
    if 'lead_time' in ds_combined.dims:
        ds_combined_subset = subset_lead_time(ds_combined, lt)
        ds_combined_actual_dates = assign_actual_dates_to_rf_dataset(ds_combined_subset)
    else:
        ds_combined_actual_dates = assign_actual_dates_to_ra_dataset(ds_combined)
    # (3) aggregate the data (using the maximum) by 'actual date'
    ds_aggregated = ds_combined_actual_dates.groupby('actual_date').max(dim = 'gauge_id')
    # (4) reset attributes and add new ones: since we're now "up" a level from gauges
    #     to units, we drop longitude, latitude, gauge_id, and admin_unit, and add the
    #     admin_unit and the gauge_id's of the gauges in the unit
    ds_aggregated.attrs = {}
    ds_aggregated.attrs['identifier'] = district_name
    ds_aggregated.attrs['gauge_ids'] = [ds.attrs['gauge_id'] for ds in datasets]

    return ds_aggregated


def aggregate_per_admin_unit(
        d_rf_datasets: Dict[str, xr.Dataset],
        d_ra_datasets: Dict[str, xr.Dataset],
        lead_time: int = 7,
        verbose: bool = True
    ) -> Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]:
    """
    Aggregate the data per administrative unit:
    - with lead time, subset the forecast horizon can be subsetted
    - verbose, whether to print some test print-s's

    At every timestep, the maximum value present over all stations in the 
    district is chosen. Thresholds are calculated using reanalysis data:
    the reanalysis data is aggregated too, and from the resulting hydrograph
    thresholds are determined.

    The process therefore consists of:
    (1) aggregating the reanalysis data;
    (2) aggregating the reforecast data;
    (3) adding the thresholds to the reforecast data (as attributes).

    Time complexity is O(n), where n is the number of admin units.

    :param d_rf_datasets: dict of reforecast datasets
    :param d_ra_datasets: dict of reanalysis datasets
    :param lead_time: lead time of the forecast to aggregate
    :param verbose: whether to print some test print-s's
    :return: dict of datasets with aggregated data, dict with return periods
    """
    # (1) aggregate the reanalysis data
    d_ra_ds_grouped = get_dict_ds_per_admin_unit(d_ra_datasets)
    print_gauge_ids_per_admin_unit(d_ra_ds_grouped) if verbose else None

    d_ra_agg_ds = {}
    for district, datasets in d_ra_ds_grouped.items():
        print(f'aggregating {district}')
        d_ra_agg_ds[district] = aggregate_district_datasets_maximally(datasets, lead_time, district)
    
    # (2) aggregate the reforecast data
    d_rf_ds_grouped = get_dict_ds_per_admin_unit(d_rf_datasets)
    print_gauge_ids_per_admin_unit(d_rf_ds_grouped) if verbose else None

    d_rf_agg_ds = {}
    for district, datasets in d_rf_ds_grouped.items():
        print(f'aggregating {district}')
        d_rf_agg_ds[district] = aggregate_district_datasets_maximally(datasets, lead_time, district)

    # (3) add the thresholds to the reforecast data
    for district, ds in d_rf_agg_ds.items():
        add_thresholds_to_dataset(ds, d_ra_agg_ds[district])
        
    return d_rf_agg_ds


def aggregate_or_load_per_admin_unit(
        d_rf: Dict[str, xr.Dataset],
        d_ra: Dict[str, xr.Dataset],
        lt: int = 7,
        verbose: bool = True,
) -> Dict[str, xr.Dataset]:
    """ Helper function to aggregate the data per gauge (with aggregate_per_admin_unit())
    to per admin unit if not done already (then load the datasets instead)

    :param d_rf: dict of datasets
    :param d_ra: dict of datasets
    :param lt: lead time of the forecast to aggregate (eg 7)
    :param verbose: whether to print some test print-s's
    :return: dict of datasets with aggregated data
    """
    if type(lt) != int or lt < 0 or lt > 7:
        raise ValueError('Lead time must be an integer between 0 and 7')
    if not os.path.exists('../data/GRRR/aggregated/'):
        os.makedirs('../data/GRRR/aggregated/', exist_ok = True)
    
    print('Checking if datasets are already loaded...')

    n_to_load = len(create_admin_unit_set(d_rf))
    n_datasets_loaded = len([f for f in os.listdir('../data/GRRR/aggregated/') if \
                             f.endswith(f'_{lt * 24}lt.nc')])
    
    if (n_to_load == n_datasets_loaded):
        print('Loading in datasets...')
        d_datasets_au = import_datasets_from_netcdf('../data/GRRR/aggregated/', lt)
        print('Loading complete')
    else:
        print('Datasets not loaded yet; aggregating...\n')
        d_datasets_au = aggregate_per_admin_unit(d_rf, d_ra, lt, verbose)
        export_datasets_to_netcdf(d_datasets_au, '../data/GRRR/aggregated/', lt)
        print('\nAggregation complete')

    return d_datasets_au


def select_lead_time(
        d_ds: Dict[str, xr.Dataset],
        lt: int = 7
) -> Dict[str, xr.Dataset]:
    """
    Selects the lead time from the datasets

    :param d_ds: dict of datasets
    :param lt: lead time to select
    :return: dict of datasets with the selected lead time
    """
    if not (0 <= lt <= 7):
        raise ValueError("Lead time must be between 0 and 7 days")

    dict_out = {}
    for key, ds in d_ds.items():
        ds_sub = ds.sel(lead_time = pd.Timedelta(days=lt))
        actual_dates = ds_sub['issue_time'] + ds_sub['lead_time']
        ds_sub = ds_sub.assign_coords(
            actual_date = ('issue_time', actual_dates.data)
        )
        ds_sub.attrs['identifier'] = key
        dict_out[key] = ds_sub

    return dict_out