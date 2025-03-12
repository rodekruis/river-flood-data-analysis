# src/analyse/statistics.py

from .events import combine_dict_events_to_df
from .transform import make_subset_for_gauge_and_issue_time

from typing import Dict
import datetime
import json
import numpy as np
import pandas as pd
import xarray as xr


def count_admin_units(dict_datasets: Dict[str, xr.Dataset]) -> int:
    """
    Count how many admin units all gauges have in total in their 'admin_unit' attribute (list)

    :param dict_datasets: dict of datasets
    :return: total number of admin units
    """
    return sum(len(ds.attrs['admin_unit']) for ds in dict_datasets.values())


def count_gauge_ids(dict_datasets: Dict[str, xr.Dataset]) -> int:
    """
    Count how many gauge IDs are in the dictionary of datasets

    :param dict_datasets: dict of datasets
    :return: total number of gauge IDs
    """
    return len(dict_datasets)


def z_normalise(series: pd.Series) -> pd.Series:
    """
    Z-normalises a series by subtracting the mean and dividing by the standard deviation

    :param series: the series to be normalised
    :return: the normalised Series
    """
    return (series - series.mean()) / series.std()


def get_stats_for_forecast_range(
        df: pd.DataFrame,
        issue_time: datetime.datetime,
        gauge_ID: str,
        delta: int,
        stat: str
    ) -> pd.Series:
    """
    Gets a dataframe with forecasts, an issue time, gauge ID, and delta
    and transform and aggregates the forecast values within that range
    into a chosen statistic, resulting in one value per date. Options:
    'min' : get minimum forecast value
    'max' : get maximum forecast value
    'mean': get mean forecast value
    'dev' : get standard deviation of forecast values
    'var' : get variance of forecast values

    :param df: dataframe containing forecasts, with for each issue time
               a seven-day forecast per gauge
    :param issue_time: issue time of the forecast to be taken
    :param delta: time delta in days from the issue time
    :param stat: which statistic to calculate
    :return: a pd.Series with the chosen statistic per date
    """
    # Check if the issue time + delta is within the range of the dataframe
    if issue_time + datetime.timedelta(days = delta + 4) > pd.to_datetime(df['fc_date'].max()):
        raise ValueError("Issue time + delta exceeds the max forecasted date in the dataframe")

    df = df.copy()
    dfs = []
    for idx in range(0, delta):
        dfs.append(make_subset_for_gauge_and_issue_time(
            df,
            gauge_ID,
            issue_time.replace(hour = 0,
                               minute = 0,
                               second = 0,
                               microsecond = 0) + datetime.timedelta(days = idx)
            )
        )

    # With the list of dfs, we now need to aggregate them such
    # that all datapoints with the same date are grouped together,
    # to then calculate the chosen statistic
    grouped = pd.concat(dfs).groupby('issue_date')['fc_value']

    if stat == 'min':
        return grouped.min()
    elif stat == 'max':
        return grouped.max()
    elif stat == 'mean':
        return grouped.mean()
    elif stat == 'dev':
        return grouped.std()
    elif stat == 'var':
        return grouped.var()
    else:
        raise ValueError(f"Statistic {stat} not recognized")
    


def calculate_metrics(d_metrics: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    Calculate the performance metrics from the metrics dictionary;
    metrics:
    - probability of detection (POD) / recall: TP / (TP + FN)
    - false alarm ratio (FAR): FP / (TP + FP)
    - precision: TP / (TP + FP)
    - f1-score: 2 * (precision * recall) / (precision + recall)

    :param d_metrics: dictionary with metrics per admin unit
    :return: dataframe with metrics
    """
    metrics_list = []       # init storages
    total_TP, total_FP, total_FN = 0, 0, 0

    for admin_unit, metrics in d_metrics.items():
        TP = metrics['TP']
        FP = metrics['FP']
        FN = metrics['FN']
        total_TP += TP      # count-based metrics
        total_FP += FP
        total_FN += FN
                            # calculate ratio's, assign NaN if denominator is zero
        POD = TP / (TP + FN) if TP + FN > 0 else np.nan
        FAR = FP / (TP + FP) if TP + FP > 0 else np.nan
        precision = TP / (TP + FP) if TP + FP > 0 else np.nan
        f1 = 2 * (precision * POD) / (precision + POD) if precision + POD > 0 else np.nan
                            # store metrics
        metrics_list.append({
            'identifier': admin_unit,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'POD': POD,	
            'FAR': FAR,
            'precision': precision,
            'f1': f1
        })
                            # calculate total metrics
    total_POD = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else np.nan
    total_FAR = total_FP / (total_TP + total_FP) if total_TP + total_FP > 0 else np.nan
    total_precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else np.nan
    total_f1 = 2 * (total_precision * total_POD) / (total_precision + total_POD) if \
        total_precision + total_POD > 0 else np.nan
                            # store total metrics at first index (0th)
    metrics_list.insert(0, {
        'identifier': 'total',
        'TP': total_TP,
        'FP': total_FP,
        'FN': total_FN,
        'POD': total_POD,
        'FAR': total_FAR,
        'precision': total_precision,
        'f1': total_f1	
    })

                            # round floats to 3 decimals 
    for metric in metrics_list:
        for key, value in metric.items():
            if isinstance(value, float):
                metric[key] = np.round(value, 3)
    df = pd.DataFrame(metrics_list).\
            sort_values('identifier').\
                reset_index(drop = True)
    # if identifier starts with 'ML', then we have admin units
    # and can add their names as a column as well
    if df['identifier'][0].startswith('ML'):
        with open('../data/mappings/PCODE_to_Cercle.json', 'r') as f:
            inverse_mapping = json.load(f)
        df['admin_unit_NAME'] = df['identifier'].map(inverse_mapping)
    
    return df


def match_events_and_get_metrics(
        d_flood_events: Dict[str, pd.DataFrame],
        d_impact_events: Dict[str, pd.DataFrame],
        action_lifetime: int = 10
) -> Dict[str, int]:
    """
    Matches flood events to impact events and calculates performance metrics;
    it calculates the following metrics per administrative unit:
    - True Positives (TP): the number of correctly predicted flood events;
    - False Positives (FP): the number of incorrectly predicted flood events; and
    - False Negatives (FN): the number of missed flood events,
    and returns them as a dictionary with admin unit as key, metrics as value.

    :param d_flood_events: dictionary with flood events
    :param d_impact_events: dictionary with impact events
    :param action_lifetime: margin of error for comparison of dates (or could
                            also be called, in a practical sense, the action lifetime)
    :return: df with total and per unit metrics (through helper function)
    """
                            # convert the event dictionaries back to dataframes
    df_flood_events = combine_dict_events_to_df(d_flood_events)
    df_impact_events = combine_dict_events_to_df(d_impact_events)
    metrics = {}
                            # get the set of unique administrative units from
                            # the flood and impact events, although they should
                            # be the same, we take the union anyway
    admin_units = set(df_flood_events['identifier']).union(set(df_impact_events['identifier']))
    for admin_unit in admin_units:
                            # zero-init metrics
        TP, FP, FN = 0, 0, 0
                            # get the flood and impact events for the admin unit
        events_pred = \
            df_flood_events[df_flood_events['identifier'] == admin_unit].reset_index(drop = True)
        events_true = \
            df_impact_events[df_impact_events['identifier'] == admin_unit].reset_index(drop = True)
                            # check if there are any events for the admin unit, if not
                            # we can skip the calculation and add metrics immediately;
                            # if no events at all, all metrics are zero
        if events_pred.empty and events_true.empty:
            metrics[admin_unit] = {'TP': TP, 'FP': FP, 'FN': FN}
            continue        # if no predicted events, all true events are false negatives
        if events_pred.empty:
            metrics[admin_unit] = {'TP': TP, 'FP': FP, 'FN': events_true.shape[0]}
            continue        # if no true events, all predicted events are false positives
        if events_true.empty:
            metrics[admin_unit] = {'TP': TP, 'FP': events_pred.shape[0], 'FN': FN}
            continue
                            # convert the events to intervals for comparison, with
                            # closed as 'both' and add margin of error for the preds:
                            # https://pandas.pydata.org/docs/reference/api/pandas.Interval.html
        intervals_pred = events_pred.apply(
            lambda row: pd.Interval(row['flood_start'] - pd.Timedelta(days = action_lifetime),
                                    row['flood_end'] + pd.Timedelta(days = action_lifetime),
                                    closed = 'both'), axis = 1)
        intervals_true = events_true.apply(
            lambda row: pd.Interval(row['flood_start'],
                                    row['flood_end'],
                                    closed = 'both'),axis = 1)
        
                            # sets to keep track of matched events and avoid double counting;
                            # loop over the predicted events and check if they
                            # match with the true events, and update the metrics
                            # (matched_pred is unused, but might be useful later)
        matched_pred = set()
        matched_true = set()
        for idx_pred, interval_pred in intervals_pred.items():
                            # set flag to False
            is_match = False
                            # look for overlap of prediction with an impact event
            for idx_imp, interval_imp in intervals_true.items():
                            # skip if event already matched (although this
                            # wouldn't be triggered in practice)
                if idx_imp in matched_true:
                    continue
                            # if overlap, update metrics and set flag to True
                if interval_pred.overlaps(interval_imp):
                    TP += 1
                    matched_pred.add(idx_pred)
                    matched_true.add(idx_imp)
                    is_match = True
                            # stop looking for impact events once a match is found;
                            # the question is whether to do this yes/no. Will discuss on Monday
                    # break
                            # no match found, so false positive is added
            if not is_match:
                FP += 1
                            # any impact events not matched are false negatives
        for idx_imp in intervals_true.index:
            if idx_imp not in matched_true:
                FN += 1
                            # store the metrics for the admin unit
        metrics[admin_unit] = {'TP': TP, 'FP': FP, 'FN': FN}

    return calculate_metrics(metrics)


def export_results(
        df: pd.DataFrame, ct: str, lt: int, th: int
    ) -> None:
    """ 
    Simple function that exports the results df in correct name format
    
    :param df: DataFrame with results
    :param ct: comparison type (e.g. 'IMPACT')
    :param lt: lead time in days (e.g. 7)
    :param th: threshold (e.g. 5)
    """
    if th in [95, 98, 99]:
        th_str = f'{th}pc'
    else:
        th_str = f'{th}rp'
    df.to_csv(
        f'../data/results/GFH_vs_{ct}_{lt * 24}lt_{th_str}.csv'
    )