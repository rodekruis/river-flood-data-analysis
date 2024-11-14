# src/analyse/statistics.py

from .transform import make_subset_for_gauge_and_issue_time

import datetime
import pandas as pd


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