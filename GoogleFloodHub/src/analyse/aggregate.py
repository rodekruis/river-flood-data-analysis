# src/analyse/aggregate.py

# Functions to aggregate data per gauge to data per administrative unit

from typing import Dict
import pandas as pd
import xarray as xr


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


def pretty_print_list(l: list) -> None:
    """
    Pretty print a list

    :param l: list
    """
    print(', '.join(l))


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


def assign_actual_dates_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Assign the actual dates to the dataset as new coordinates

    :param ds: xarray Dataset
    :return: xarray Dataset with actual dates as coordinates
    """
    actual_dates = ds['issue_time'] + ds['lead_time']
    return ds.assign_coords(actual_date = ('issue_time', actual_dates.data))


# def aggregate_per_admin_unit_N_squared(
#         dict_datasets: Dict[str, xr.Dataset],
#         lead_time: int = 7,
#         method: str = 'max',
#         verbose: bool = True
#     ) -> Dict[str, xr.Dataset]:
#     """
#     EDIT: this is the old version with time complexity O(n ^ 2),
#     which proved too slow. aggregate_per_admin_unit() is the new onr
#     Aggregate the data per administrative unit with a method
#     of choice, defaulting to 'max'; more options to be added later.
#     With lead time, the forecast horizon can be subsetted

#     :param dict_datasets: dict of datasets
#     :param lead_time: lead time of the forecast to aggregate
#     :param method: method of aggregation
#     :param verbose: whether to print some test print-s's
#     :return: dict of datasets with aggregated data
#     """
#     # create a unique set of admin units
#     admin_units = create_admin_unit_set(dict_datasets)
#     pretty_print_list(admin_units) if verbose else None
#     dict_datasets_aggregated = {}

#     # for every admin unit: (1) create a list of datasets for that unit;
#     # this has a time complexity of O(n ^ 2), but n is quite small
#     idx = 1
#     for admin_unit in admin_units:
#         if verbose:
#             print(f'aggregating {idx}/{len(admin_units)}: {admin_unit}')
#             idx += 1
#         datasets_admin_unit = [
#             ds for ds in dict_datasets.values() if \
#                 admin_unit in ds.attrs.get('admin_unit', [])
#         ]
#         if not datasets_admin_unit:
#             print(f'No datasets found for admin unit {admin_unit}')
#             continue
#         # (2) concatenate the datasets into one dataset and add gauge_id dimension;
#         # (3) filter by lead time, discarding the other lead times; (4) assign the
#         # actual dates to the dataset, a.k.a. the date at which the forecast actually
#         # applies to; (5) aggregate the data by 'actual date' and calculate with 'method'
#         ds_combined = xr.concat(datasets_admin_unit, dim = 'gauge_id')
#         ds_combined_subset = subset_lead_time(ds_combined, lead_time)
#         ds_combined_actual_dates = assign_actual_dates_to_dataset(ds_combined_subset)
        
#         if method == 'max':
#             ds_aggregated = \
#                 ds_combined_actual_dates.groupby('actual_date').max(dim = 'gauge_id')
#         elif method == 'mean':
#             ds_aggregated = \
#                 ds_combined_actual_dates.groupby('actual_date').mean(dim = 'gauge_id')
#         else:
#             raise ValueError('Method parameter not recognised')

#         # lastly, we update the attributes of the dataset of the admin unit,
#         # since now we're up a level from gauges to units, asking for a replacement
#         # of the attributes: we drop longitude, latitude, gauge_id, and admin_unit,
#         # and add the admin_unit and the gauge_id's of the gauges in the unit
#         ds_aggregated.attrs = {'admin_unit': admin_unit,
#                                'gauge_ids': [ds.attrs['gauge_id'] for ds in datasets_admin_unit]}
#         dict_datasets_aggregated[admin_unit] = ds_aggregated
    
#     return dict_datasets_aggregated


def get_dict_ds_per_admin_unit(dict_ds: Dict[str, xr.Dataset]) -> Dict[str, list]:
    """
    Get a dictionary with the datasets per admin unit

    :param dict_ds: dict of datasets
    :return: dict of datasets per admin unit
    """
    admin_units = create_admin_unit_set(dict_ds)
    dict_ds_per_admin_unit = {unit: [] for unit in admin_units}
    for ds in dict_ds.values():
        for unit in ds.attrs['admin_unit']:
            dict_ds_per_admin_unit[unit].append(ds)
    return dict_ds_per_admin_unit


# def aggregate_per_admin_unit(
#         dict_datasets: Dict[str, xr.Dataset],
#         lead_time: int = 7,
#         method: str = 'max',
#         verbose: bool = True
#     ) -> Dict[str, xr.Dataset]:
#     """
#     Aggregate the data per administrative unit with a method
#     of choice, defaulting to 'max'; more options to be added later.
#     With lead time, the forecast horizon can be subsetted

#     :param dict_datasets: dict of datasets
#     :param lead_time: lead time of the forecast to aggregate
#     :param method: method of aggregation
#     :param verbose: whether to print some test print-s's
#     :return: dict of datasets with aggregated data
#     """
#     # get the datasets per admin unit for aggregation next;
#     # check which admin units did not get any dataset assigned
#     grouped_datasets = get_dict_ds_per_admin_unit(dict_datasets)
#     if verbose:
#         print("[admin unit ID] : list([gauge ID's])")
#         for unit, datasets in grouped_datasets.items():
#             print(unit, end = ' : ')
#             pretty_print_list([ds.attrs['gauge_id'] for ds in datasets])
#         print('\n')

#     dict_datasets_aggregated = {}
#     # time complexity is O(n), where n is the number of admin units
#     idx = 1
#     for admin_unit, datasets in grouped_datasets.items():
#         if verbose:
#             print(f'aggregating {idx}/{len(grouped_datasets)}: {admin_unit}')
#             idx += 1
        
#         # (2) concatenate the datasets into one dataset and add gauge_id dimension;
#         # (3) filter by lead time, discarding the other lead times; (4) assign the
#         # actual dates to the dataset, a.k.a. the date at which the forecast actually
#         # applies to; (5) aggregate the data by 'actual date' and calculate with 'method'
#         ds_combined = xr.concat(datasets, dim = 'gauge_id')
#         ds_combined_subset = subset_lead_time(ds_combined, lead_time)
#         ds_combined_actual_dates = assign_actual_dates_to_dataset(ds_combined_subset)
        
#         if method == 'max':
#             ds_aggregated = \
#                 ds_combined_actual_dates.groupby('actual_date').max(dim = 'gauge_id')
#         elif method == 'mean':
#             ds_aggregated = \
#                 ds_combined_actual_dates.groupby('actual_date').mean(dim = 'gauge_id')
#         else:
#             raise ValueError('Method parameter not recognised')

#         # lastly, we update the attributes of the dataset of the admin unit,
#         # since now we're up a level from gauges to units, asking for a replacement
#         # of the attributes: we drop longitude, latitude, gauge_id, and admin_unit,
#         # and add the admin_unit and the gauge_id's of the gauges in the unit
#         ds_aggregated.attrs = {'admin_unit': admin_unit,
#                                'gauge_ids': [ds.attrs['gauge_id'] for ds in datasets]}
#         dict_datasets_aggregated[admin_unit] = ds_aggregated
    
#     return dict_datasets_aggregated