# src/extract_data/plots.py

from extract import get_json_file

from .getters import get_country_polygon
from .getters import get_severity_levels
from .transform import convert_df_to_gdf
from .transform import make_subset_for_gauge_and_issue_time
from .transform import convert_country_code_to_iso_a3
from .statistics import z_normalise
from .statistics import get_stats_for_forecast_range

from typing import Any, List, Tuple
import datetime
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def create_custom_palette(n_colours) -> list:
    """
    Returns a custom palette with red and dark blue colors

    :param n_colours: number of colours in the palette
    :return: list with n colours
    """
    cmap = LinearSegmentedColormap.from_list("red_to_blue",
                                             ['#DB0A13', '#092448'],
                                             N = n_colours)
    return [cmap(idx) for idx in range(cmap.N)]


def set_TeX_style() -> None:
    """
    Sets the style environment for the plots to use TeX for text rendering
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "axes.labelsize": 10,     # fontsize for x and y labels
        "xtick.labelsize": 10,    # fontsize of the tick labels
        "ytick.labelsize": 10,    # fontsize of the tick labels
        "legend.fontsize": 10,    # fontsize of the legen
    })


def set_plot_style(
        TeX: bool = False, style: str = 'ticks', context: str = None) -> None:
    """
    Sets the style environment for the plots, including:
    - through the set_TeX_style function:
        - whether to use TeX for text rendering
        - the font family for titles, labels, etc.
        - the font size for titles, labels, etc.
    - seaborn style to use (default = 'ticks')
    - the context for the plot (default = 'notebook')
    
    :param TeX: whether to use TeX for text rendering
    :param style: seaborn style to use
    :param context: the context for the plot
    """
    if TeX:
        set_TeX_style()
    sns.set_theme(style = 'ticks')
    sns.set_context(context) if context else None


def create_dates_list(start_date : datetime.datetime, delta: int) -> List[datetime.datetime]:
    """
    Create a series of dates starting from a given date and going on for a given number of days

    :param start_date: starting date
    :param delta: number of days
    :return: list with dates
    """
    return [
        # minus one here because, strangely enough, the first forecasted
        # date is one day in the past compared to the issue date...
        start_date + datetime.timedelta(days = idx - 1) for idx in range(0, delta + 1)
    ]


def set_custom_date_ticks(
        ax: plt.Axes, issue_date: datetime.datetime, days: int) -> None:
    """
    Set custom date ticks on the x-axis of a plot, where only
    the first and final date are displayed, while keeping the ticks

    :param ax: axes object
    :param dates: list of dates (datetime.datetime objects)
    """
    dates = create_dates_list(issue_date, days + 7)
    if len(dates) < 2:
        raise ValueError('At least two dates are needed to set custom date ticks')

    ax.set_xticks(mdates.date2num(dates))
    x_labels = [
        dates[0].strftime('%Y-%m-%d')
    ] + [''] * (len(dates) - 2) + [
        dates[-1].strftime('%Y-%m-%d')
    ]
    ax.set_xticklabels(x_labels)


def plot_gauge_forecast_for_issue_time(
        df : pd.DataFrame, gauge: str, issue_date : datetime.datetime, country : str = None) -> None:
    """
    Plots with a graph the forecasted values for a specific gauge and issue time

    :param df: DataFrame with forecasted values
    :param gauge: ID of the gauge
    :param issue_time: issue time of the forecast
    :param country: name of the country
    """
    set_plot_style()

    # for plotting purposes, the time can be normalized
    issue_date = issue_date.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    df_subset = make_subset_for_gauge_and_issue_time(df, gauge, issue_date)
    if df_subset.empty:
        print(f"No forecasted values for gauge {gauge} at {issue_date.date()}")
        return
    df_subset['fc_date'] = mdates.date2num(df_subset['fc_date'])

    ax = sns.lineplot(
        x = 'fc_date',
        y = 'fc_value',
        data = df_subset,
        color = '#DB0A13'
    )

    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # only display first and final forecast date on date-axis while keeping ticks
    set_custom_date_ticks(ax, issue_date, 7)

    plt.title(f'Forecasts for gauge {gauge} in {country}, w/ 1st issue date: {issue_date.date()}')
    plt.xlabel('date')
    plt.ylabel('[m³/s]')
    plt.show()


def plot_week_of_gauge_forecast_for_issue_time(
        df : pd.DataFrame, gauge: str, issue_date : datetime.datetime, country : str = None) -> None:
    """
    Plots the forecasted values for a specific gauge over a week of
    issue times, giving seven graphs in total, each of (7 + 1 =) 8 days length 

    :param df: DataFrame with forecasted values
    :param gauge: ID of the gauge
    :param issue_time: first issue time
    :param country: Name of the country
    """
    set_plot_style()
    plt.figure(figsize = (10, 6))
    custom_palette = create_custom_palette(7)

    # for plotting purposes, the time can be normalized
    issue_date = issue_date.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    
    for idx in range(7): # loop for seven days, aka a week
        df_subset = make_subset_for_gauge_and_issue_time(
            df, gauge, issue_date + datetime.timedelta(days = idx)
        ).copy()
        if df_subset.empty:
            print(f"No forecasted values for gauge {gauge} at {issue_date.date()}")
            return
        df_subset['fc_date'] = mdates.date2num(df_subset['fc_date'])

        sns.lineplot(
            x = 'fc_date',
            y = 'fc_value',
            data = df_subset,
            color = custom_palette[idx]
        )
    # plt.gca() returns the current axes, 7 + 7 = 14 days = a week plus lead time
    set_custom_date_ticks(plt.gca(), issue_date, 7 + 7)
    

    plt.title(f'Forecasts for gauge {gauge} in {country}, w/ 1st issue date: {issue_date.date()}')
    plt.xlabel('date')
    plt.ylabel('[m³/s]')
    plt.show()


def plot_x_days_of_gauge_forecast_for_issue_time(
        df_forecasts : pd.DataFrame,
        df_gauge_models: pd.DataFrame,
        gauge: str,
        issue_date : datetime.datetime,
        days: int,
        country : str = None,
        TeX: bool = False,
        export: bool = False) -> None:
    """
    Plots the forecasted values for a specific gauge over a month of
    issue times, giving 30 graphs in total, each of (7 + 1 =) 8 days length 

    :param df: DataFrame with forecasted values
    :param df_gauge_models: DataFrame with gauge models
    :param gauge: ID of the gauge
    :param issue_date: first issue time
    :param country: Name of the country
    :param TeX: whether to use TeX for text rendering
    :param export: whether to export the plot as a pdf
    """
    set_plot_style()
    set_TeX_style() if TeX else None
    plt.figure(figsize = (10, 6))
    custom_palette = create_custom_palette(days)

    # for plotting purposes, the time can be normalized
    issue_date = issue_date.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

    # plot severity levels (or equivalently, return period values)
    severity_levels = get_severity_levels(df_gauge_models, gauge)
    plt.axhline(y = severity_levels['two_year'], color = 'green',
                linestyle = '--', linewidth = 0.8, label = '2-year rp')
    plt.axhline(y = severity_levels['five_year'], color = 'orange',
                linestyle = '--', linewidth = 0.8, label = '5-year rp')
    plt.axhline(y = severity_levels['twenty_year'], color = 'red',
                linestyle = '--', linewidth = 0.8, label = '20-year rp')
    
    for idx in range(days):
        df_subset = make_subset_for_gauge_and_issue_time(
            df_forecasts, gauge, issue_date + datetime.timedelta(days = idx)
        ).copy()
        if df_subset.empty:
            print(f"No forecasted values for gauge {gauge} at {issue_date.date()}")
            return
        # for x-axis (type) alignment
        df_subset['fc_date'] = mdates.date2num(df_subset['fc_date'])

        sns.lineplot(
            x = 'fc_date',
            y = 'fc_value',
            data = df_subset,
            color = custom_palette[idx]
        )

    # plt.gca() returns the current axes, 7 + days = lead time + delta days
    set_custom_date_ticks(plt.gca(), issue_date, days + 7)

    plt.title(f'Forecasts for gauge {gauge} in {country}, w/ 1st issue date: {issue_date.date()}')
    plt.xlabel('date')
    plt.ylabel('[m³/s]')
    plt.draw()

    if export:
        plt.savefig(f"../plots/graph_{days}_fcs_of_gauge_{gauge}_at_issue_date_{str(issue_date.date())}.pdf",
                    format = 'pdf',
                    bbox_inches = 'tight',
                    pad_inches = 0.015)
        
    plt.show()


def plot_danger_levels_hist(df: pd.DataFrame, country: str = None, bins: int = 10) -> None:
    """
    Plot histogram of danger levels in the dataset

    :param df: DataFrame with gauge information
    """
    set_plot_style()

    risk_levels = ['dangerLevel', 'extremeDangerLevel', 'warningLevel']
   
    for level in risk_levels:
        ax = sns.histplot(pd.DataFrame(df[level]),
                          color = '#DB0A13',
                          bins = bins)
        
        for patch in ax.patches:
            patch.set_facecolor('#DB0A13')
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer = True))
        plt.legend(handles = [Patch(color = '#DB0A13', label = level)])
        plt.xlabel('Cubic meter per second')
        plt.ylabel('Frequency')
        plt.title('Distribution of danger levels in ' + country)
        
        plt.show()


def map_gauge_coordinates_of_country(df : pd.DataFrame, country : str) -> None:
    """
    Map gauge coordinates

    :param df: the DataFrame with the gauges (so ListGauges API call)
    :param country: the country of interest
    :return None
    """
    gdf = convert_df_to_gdf(df)
    shape = get_country_polygon(get_json_file(
        "../data/country_code_conversions/country_codes.json"
        )[country])

    fig, ax = plt.subplots()
    shape.plot(ax = ax, color = 'lightgrey')
    gdf.plot(ax = ax, color = 'red', markersize = 10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Gauge locations for {country}')
    ax.set_aspect('equal') # Ensure unwarped aspect ratio

    plt.show()


def plot_forecast_min_mean_max(
        df: pd.DataFrame,
        issue_date: datetime.datetime,
        gauge: str,
        delta: int) -> None:
    """
    Plots the minimum, mean, and maximum forecast values
    for a given gauge and issue time, for a given number of days
    after the issue time

    :param df: dataframe containing forecasts, with for each issue time
               a seven-day forecast per gauge
    :param issue_date: issue date/time of the forecast to be taken
    :param gauge: gauge ID for which to plot the forecast
    :param delta: time delta in days from the issue time
    :return None
    """
    set_plot_style()
    plt.figure(figsize = (10, 6))
    custom_palette = create_custom_palette(3)
    
    min_values = get_stats_for_forecast_range(df, issue_date, gauge, delta, 'min')
    mean_values = get_stats_for_forecast_range(df, issue_date, gauge, delta, 'mean')
    max_values = get_stats_for_forecast_range(df, issue_date, gauge, delta, 'max')

    sns.lineplot(
        x = min_values.index,
        y = min_values,
        label = 'min',
        color = custom_palette[0]
    )
    sns.lineplot(
        x = mean_values.index,
        y = mean_values,
        label = 'mean',
        color = custom_palette[1]
    )
    sns.lineplot(
        x = max_values.index,
        y = max_values,
        label = 'max',
        color = custom_palette[2]
    )

    plt.xticks(rotation = 45)
    plt.title(f"Forecast for gauge {gauge} issued on {str(issue_date)[:10]}")
    plt.xlabel("Date")
    plt.ylabel("Cubic meters per second")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_Niger_river_downstream_flow_stat(
        df: pd.DataFrame,
        issue_date: datetime.datetime,
        gauges: list,
        delta: int,
        stat) -> None:
    """
    Plots a statistic of choice for the gauges along the Niger river
    in Mali in downstream order. The statistic is calculated for a
    given number of days after the issue time. Each gauge has its
    volume normalised to account for the different local volumes

    :param df: dataframe containing forecasts, with for each issue time
                a seven-day forecast per gauge
    :param issue_date: issue date/time of the forecast to be taken
    :param gauges: list of gauge IDs for which to plot the forecast
    :param delta: time delta in days from the issue time
    :param stat: which statistic to calculate
    """
    set_plot_style()
    plt.figure(figsize = (10, 6))
    custom_palette = create_custom_palette(len(gauges))

    for idx, gauge in enumerate(gauges):
        values = z_normalise(
            get_stats_for_forecast_range(
                df, issue_date, gauge, delta, stat
            )
        )
        sns.lineplot(
            x = values.index,
            y = values,
            label = gauge,
            color = custom_palette[idx]
        )
    
    plt.xticks(rotation = 45)
    plt.title(f"z-normalised {stat} forecast for gauges along the Niger river in Mali")
    plt.xlabel("date")
    plt.ylabel("z-normalised volume")
    plt.legend()
    plt.tight_layout()
    plt.show()


def add_return_periods(
        ax: mpl.axes, gauge_return_periods_ds: xr.core.dataset.Dataset,
        thresholds: Tuple[int], set_legend: bool = True
    ) -> None:
    """
    Adds horizontal lines with return periods to a plot

    :param ax: axis to add the return periods to
    :param gauge_return_periods_ds: return periods dataset
    :param thresholds: list of thresholds to add return periods for
    """
    colors = ('yellow', 'orange', 'red', 'brown', 'black')

    # if there is only one colour, set it to orange
    # add legend, make a legend with a line 
    for threshold, color in zip(thresholds, colors):
        ax.axhline(
            y = gauge_return_periods_ds[f'return_period_{threshold}'].item(),
            color = color if len(thresholds) > 1 else 'orange',
            label = f'{threshold}-yr return period' if set_legend else None,
            linestyle = '--'
        )


def plot_reforecast(
        issue_time_start_date: str, issue_time_end_date: str,
        ds_reforecast: xr.core.dataset.Dataset,
        ds_return_periods: xr.core.dataset.Dataset,
        thresholds: Tuple[int] = ('2', '5', '20')
    ) -> None:
    """ 
    Plots reforecast data for a give time range

    :param issue_time_start_date: start date for the issue time
    :param issue_time_end_date: end date for the issue time
    :param gauge_return_periods_ds: return periods dataset
    :param thresholds: list of thresholds to add return periods for
    """
    fig, ax = plt.subplots(figsize = (20, 4))
    issue_times = ds_reforecast.sel(issue_time = \
                        slice(issue_time_start_date, issue_time_end_date))['issue_time']

    for issue_time in issue_times:      # select issue time slice
        issue_time_slice = ds_reforecast.sel(issue_time = issue_time)
                                        # convert lead time to dates and
                                        # divide by 10**9 to convert to seconds
        lead_time_to_dates = [pd.to_datetime(issue_time.values) + \
                              datetime.timedelta(seconds = (lead_time.item() // 10**9)) \
                                for lead_time in issue_time_slice['lead_time']]
                                        # for each lead time, plot the streamflow
        ax.plot(lead_time_to_dates, issue_time_slice.streamflow.values)

    add_return_periods(ax, ds_return_periods, thresholds)
    plt.legend(loc = 'upper right')
    plt.show()


def plot_reanalysis(start_date: str, end_date: str,
                    ds_reanalysis: xr.core.dataset.Dataset,
                    ds_return_periods: xr.core.dataset.Dataset,
                    thresholds = ('2', '5', '20')
    ) -> None:
    """
    Plots reanalysis data for a given time range

    :param start_date: start date for the time range
    :param end_date: end date for the time range
    :param gauge_reanalysis_ds: reanalysis dataset
    :param gauge_return_periods_ds: return periods dataset
    :param thresholds: list of thresholds to add return periods for
    """
    fig, ax = plt.subplots(figsize = (20, 4))
    ds_subset = ds_reanalysis.streamflow.sel(time = slice(start_date, end_date))
    
    ax.plot(ds_subset.time, ds_subset.values)
    add_return_periods(ax, ds_return_periods, thresholds)
    plt.legend(loc = 'upper right')
    plt.show()