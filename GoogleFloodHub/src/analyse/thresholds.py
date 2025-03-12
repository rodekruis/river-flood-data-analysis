# src/analyse/thresholds.py

from .tests import assure_all_thresholds_added

from typing import Any, List, Dict
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from pyextremes import EVA


# Functions used for calculating thresholds, i.e.
# return period thresholds and percentile thresholds


def calculate_annual_max(ds: xr.Dataset) -> xr.DataArray:
    """
    Given a reanalysis dataset, it computes the annual maximum
    discharge values for the dataset

    :param ds: the reanalysis dataset
    :return: the dataset with annual maximum values
    """
    # exclude non-complete years from the reanalysis dataset, because
    # these might not include the rain season, and, consequently, neither
    # the annual water rises and/or floodings, and thus the annual maxima
    # ... this subsetting procedure is moved elsewhere

    if not pd.api.types.is_datetime64_any_dtype(ds['time']):
        ds['time'] = pd.to_datetime(ds['time'].values)
    return ds['streamflow'].resample(time = 'Y').max()


def calculate_RP_Gumbel(ds: xr.Dataset, RP: float) -> float:
    """
    Given a reanalysis dataset and a return period (in years),
    it calculates the discharge value that corresponds to the
    given return period using a Gumbel distribution fit

    :param ds: the reanalysis dataset
    :param RP: the return period in years
    :return: the discharge value corresponding to the return period
    """                     
    annual_max = calculate_annual_max(ds).to_series()
    # print(annual_max)
                            # fit a Gumbel distribution to maxima
    loc, scale = stats.gumbel_r.fit(annual_max)
    return stats.gumbel_r.ppf(1 - (1 / RP), loc = loc, scale = scale)


def calculate_RP_LogPearsonIII(ds: xr.Dataset, RP: float) -> float:
    """ 
    Given a reanalysis dataset and a return period (in years),
    it calculates the discharge value that corresponds to the
    given return period using a Log-Pearson Type-III dist. fit

    :param ds: the reanalysis dataset
    :param RP: the return period in years
    :return: the discharge value corresponding to the return period
    """                     
    annual_max = calculate_annual_max(ds).to_series()
                            # log-transform the maxima and fit
                            # a Log-Pearson Type-III distribution
    shape, loc, scale = stats.pearson3.fit(np.log10(annual_max))
                            # calculate z-value for the given return period,
                            # and then the return period value by inverting
    return 10 ** stats.pearson3.ppf(1 - (1 / RP), shape, loc = loc, scale = scale)


def calculate_RP_GEV(ds: xr.Dataset, RP: float) -> float:
    """ 
    Given a reanalysis dataset and a return period (in years),
    it calculates the discharge value that corresponds to the
    given return period using a Generalized Extreme Value dist. fit

    :param ds: the reanalysis dataset
    :param RP: the return period in years
    :return: the discharge value corresponding to the return period
    """                     
    annual_max = calculate_annual_max(ds).to_series()
                            # fit a Generalized Extreme Value distribution to maxima
    loc, scale, shape = stats.genextreme.fit(annual_max)
                            #! Deze werkt niet goed, niet gebruiken
    return stats.genextreme.ppf(1 - (1 / RP), loc = loc, scale = scale, c = shape)


def calculate_RP_EVA(ds: xr.Dataset, RP: Any, method: str = 'GEV') -> float:
    """ 
    Given a reanalysis dataset and a return period (in years),
    it calculates the discharge value for the RP using the
    pyextremes.eva.EVA package for a method of choice
    
    URL: https://georgebv.github.io/pyextremes/api/eva/

    :param ds: the reanalysis dataset
    :param RP: the return period(s) in years, as float or array
    :param method: the method to use for the calculation
    :return: the discharge value corresponding to the return period
    """
    model = EVA(ds['streamflow'].to_series())

    if method == 'GEV':    # Generalized Extreme Value distribution with:
                           # block method for block maxima, default freq is annual
        model.get_extremes(method = 'BM')
        model.fit_model(distribution = 'genextreme')
    else:
        raise ValueError(f'Unknown method: {method}')
                            #! Deze werkt ook niet goed, niet gberuiken
    return_value = model.get_return_value(return_period = RP)

    return return_value[0]


def add_return_periods_to_dataset(
        ds_rf: xr.Dataset,
        ds_ra: xr.Dataset,
        method: str = 'Gumbel',
        rps: List[str] = ['1.5', '2', '5', '7', '10', '20']
    ) -> xr.Dataset:
    """
    Adds return period values to the dataset for the given return periods

    :param ds_rf: the (reforecast) dataset to add return periods to
    :param ds_ra: the (reanalysis) dataset to use for the calculations
    :param method: the method to use for the calculation: either 
                   'Gumbel', 'Log-Pearson III', 'GEV', or 'EVA'
    :param rps: the return periods to calculate, default is
                [1.5, 2, 5, 7, 10, 20]
    """
    if method not in ['Gumbel', 'Log-Pearson III', 'GEV', 'EVA']:
        raise ValueError(f'Unknown method: {method}')
    if method == 'Gumbel':
        for rp in rps:
            ds_rf.attrs[f'RP_{rp}'] = calculate_RP_Gumbel(ds_ra, float(rp))
    elif method == 'Log-Pearson III':
        for rp in rps:
            ds_rf[f'RP_{rp}'] = calculate_RP_LogPearsonIII(ds_ra, float(rp))
    elif method == 'GEV':
        for rp in rps:
            ds_rf[f'RP_{rp}'] = calculate_RP_GEV(ds_ra, float(rp))
    elif method == 'EVA':
        for rp in rps:
            ds_rf[f'RP_{rp}'] = calculate_RP_EVA(ds_ra, float(rp))
    return ds_rf


def add_percentiles_to_dataset(
        ds_receiver: xr.Dataset,
        ds_provider: xr.Dataset,
        percentiles: List[int] = [95, 98, 99]
) -> xr.Dataset:
    """
    Adds percentile values to the dataset for the given percentiles

    :param ds_receiver: the (reforecast) dataset to add percentiles to
    :param ds_provider: the (reanalysis) dataset to use for the calculations
    :param percentiles: the percentiles to calculate, default is
                        [95, 98, 99]
    """
    for perc in percentiles:
        ds_receiver.attrs[f'pc_{perc}th'] = np.percentile(ds_provider['streamflow'], perc)
    return ds_receiver
    

def get_common_IDs(
        dict_rf: Dict[str, xr.Dataset],
        dict_ra: Dict[str, xr.Dataset]
    ) -> List[str]:
    """
    Returns the common IDs between the two dictionaries

    :param dict_rf: dictionary with reforecast datasets
    :param dict_ra: dictionary with reanalysis datasets
    :return: the list of common IDs
    """
    # extract only the ID part from the keys, e.g.
    # ds_reforecast_hybas_1120724680 -> 1120724680; and
    # ds_reanalysis_hybas_112072468 -> 1120724680, to match IDs
    rf_IDs = [key.split('_')[-1] for key in dict_rf.keys()]
    ra_IDs = [key.split('_')[-1] for key in dict_ra.keys()]
    return list(set(rf_IDs).intersection(set(ra_IDs)))


def add_thresholds_to_dataset(
        ds_receiver: xr.Dataset,
        ds_provider: xr.Dataset,
        method: str = 'Gumbel',
        rps: List[str] = ['1.5', '2', '5', '10'],
        percentiles: List[int] = [95, 98, 99]
) -> xr.Dataset:
    """ Adds thresholds to a given dataset, calculated on another dataset.
    These can, of course, be the same one (e.g. reanalysis to reanalysis).

    :param ds_receiver: the dataset to add thresholds to
    :param ds_provider: the dataset to use for the calculations
    :param method: the method to use for the calculation
    :param rps: the return periods to calculate
    :param percentiles: the percentiles to calculate
    :return: the updated dataset
    """
    ds_receiver = add_return_periods_to_dataset(
        ds_receiver, ds_provider, method, rps
    )
    ds_receiver = add_percentiles_to_dataset(
        ds_receiver, ds_provider, percentiles
    )
    return ds_receiver


def add_RPs_and_percentiles(
        dict_rf: Dict[str, xr.Dataset],
        dict_ra: Dict[str, xr.Dataset],
        rps: List[str] = ['1.5', '2', '5', '7', '10', '20'],
        percentiles: List[int] = [95, 98, 99],
        method: str = 'Gumbel',
        verbose = True
) -> Dict[str, xr.Dataset]:
    """
    Helper function to add return periods and percentiles to all datasets

    :param dict_rf: dictionary with reforecast datasets
    :param dict_ra: dictionary with reanalysis datasets
    :param rps: the return periods to calculate
    :param percentiles: the percentiles to calculate
    :param method: the method to use for the calculation
    :param verbose: whether to print messages
    :return: the updated dictionary with all datasets
    """
    common_keys = get_common_IDs(dict_rf, dict_ra)
    
    if len(common_keys) == 0:
        raise ValueError('No common keys found between the dictionaries')
    if len(common_keys) != len(dict_rf) or len(common_keys) != len(dict_ra):
        raise ValueError('Dictionaries have different lengths')

                            # add the RPs and %'s
    idx = 1
    for key in common_keys:
        if verbose:
            if idx % 120 == 0:
                print(f'{idx/len(common_keys)*100:.2f}% done')
            idx += 1

        ds_rf = dict_rf[f'ds_{key}']
        ds_ra = dict_ra[f'ds_{key}']
                            # add to reforecast dict datasets
        if rps:
            dict_rf[f'ds_{key}'] = add_return_periods_to_dataset(
                ds_rf, ds_ra, method, rps
            )
        if percentiles:
            dict_rf[f'ds_{key}'] = add_percentiles_to_dataset(
                ds_rf, ds_ra, percentiles
            )
                            # check if all attributes were added
    assure_all_thresholds_added(dict_rf, common_keys, rps, percentiles)

    print('100% done')
    return dict_rf