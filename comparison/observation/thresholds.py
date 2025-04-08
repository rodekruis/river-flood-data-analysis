import pandas as pd
import GloFAS.GloFAS_prep.configuration as cfg
import pyextremes as pe
from pyextremes import EVA
from scipy.stats import genextreme, gumbel_r
import matplotlib.pyplot as plt
from scipy.stats import probplot
import matplotlib.pyplot as plt
import numpy as np

def calculate_max_discharge(hydro_df: pd.DataFrame, 
                            value_col: str = 'Value', 
                            timescale: str = 'Year', 
                            incomplete_lastyear: (str,int) = 2023, # not considering the maximum of incomplete years, as these are not representative of the season
                            date_col: str ='Date') -> pd.Series:
    """
    Calculates the maximum discharge values for a given timescale.

    Parameters:
        hydro_df (pd.DataFrame): DataFrame with discharge data. The index must be datetime or convertible to datetime.
        value_col (str): The column name of the discharge values.
        timescale (str): 'Year' for annual maxima, 'Day' for daily maxima.

    Returns:
        pd.Series: Maximum discharge values with the corresponding timescale as the index.
    """
    if not isinstance(hydro_df.index, pd.DatetimeIndex):
        hydro_df [date_col] = hydro_df.index
        hydro_df.index = pd.to_datetime(hydro_df.index)

    hydro_df = hydro_df.copy()  
    if hydro_df.index.isnull().any():
        raise ValueError("Some index entries could not be converted to datetime.")
    hydro_df = hydro_df.loc[hydro_df.index < pd.to_datetime(f'{incomplete_lastyear}-01-01')]

    if value_col not in hydro_df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame.")
    
    hydro_df = hydro_df [[f'{value_col}']]

    hydro_df = (
        hydro_df
        .sort_index(ascending=True)
        .astype(float)
        .dropna()
        )
    if timescale == 'Year':
        hydro_df['Year'] = hydro_df.index.year

        maximum =  hydro_df.groupby('Year').max()
    elif timescale == 'Day':
        hydro_df['Day_Date'] = hydro_df.index.date
        maximum = hydro_df.groupby('Day_Date').max()
    else:
        raise ValueError("Invalid timescale. Use 'Year' or 'Day'.")

    return maximum
    
def csv_to_cleaned_series(csv: str) -> pd.Series:
    """
    Reads a CSV file and prepares a cleaned pandas Series for analysis.

    :param csv: Path to the CSV file containing discharge data.
    :return: A cleaned pandas Series of discharge values.
    """
    df = pd.read_csv(
        csv,
        index_col=0,  # setting ValidTime as index column
        parse_dates=True,  # parsing dates in ValidTime
    )
    df['percentile_40.0'] = df['percentile_40.0'].interpolate(method='time')
    series = df['percentile_40.0']
    series = series.loc[series.index < pd.to_datetime('2023-01-01')]
    series = (
        series
        .sort_index(ascending=True)
        .astype(float)
        .dropna()
    )
    return series

def Q_GEV_fit(hydro_df: pd.DataFrame, value_col: str = 'Value', timescale='Year') -> tuple:
    """
    Fits a GEV distribution to the annual maximum discharge values and calculates the discharge 
    corresponding to a given percentile.

    Parameters:
        hydro_df (pd.DataFrame): DataFrame with discharge data. Index must be datetime.
        percentile (float): Percentile to calculate (0-100).
        value_col (str): The column name for discharge values.

    Returns:
        shape, loc, scale: tuple of corresponding fit
    """
    # Calculate annual maximum discharges
    daily_max_discharge = calculate_max_discharge(hydro_df, value_col, timescale=timescale)

    # Fit a GEV distribution
    shape_GEV, loc_GEV, scale_GEV = genextreme.fit(daily_max_discharge)
    return shape_GEV, loc_GEV, scale_GEV

def Q_GEV_fit_percentile (hydro_df: pd.DataFrame, percentile: float, value_col: str = 'Value') -> float:
    shape,loc,scale = Q_GEV_fit (hydro_df, value_col, timescale='Day')
    # Calculate the discharge value corresponding to the percentile
    discharge_value = genextreme.ppf(percentile / 100, shape, loc=loc, scale=scale)
    return discharge_value


def Q_Gumbel_fit_percentile(hydro_df: pd.DataFrame, percentile: float, value_col: str = 'Value', date_col='Date') -> float:
    """
    Fits a Gumbel distribution to the annual maximum discharge values and calculates the discharge 
    corresponding to a given percentile.

    Parameters:
        hydro_df (pd.DataFrame): DataFrame with discharge data. Index must be datetime.
        percentile (float): Percentile to calculate (0-100).
        value_col (str): The column name for discharge values.

    Returns:
        float: The discharge value corresponding to the given percentile.
    """
    # Calculate annual maximum discharges
    daily_max_discharge = calculate_max_discharge(hydro_df, value_col, timescale='Day', date_col=date_col)

    # Fit a Gumbel distribution
    loc, scale = gumbel_r.fit(daily_max_discharge)
    
    # Calculate the discharge value corresponding to the percentile
    discharge_value = gumbel_r.ppf(percentile / 100, loc, scale)
    return discharge_value

def Q_Gumbel_fit (hydro_df: pd.DataFrame, 
                    value_col: str = 'Value', 
                    timescale: str = 'Year',
                    date_col: str='Date'):
    max_discharges = calculate_max_discharge(hydro_df, value_col, timescale=timescale, date_col=date_col)
    loc_gumbel, scale_gumbel = gumbel_r.fit(max_discharges)
    return loc_gumbel, scale_gumbel

def Q_Gumbel_fit_RP(hydro_df: pd.DataFrame, 
                    RP: float, 
                    value_col: str = 'Value',
                    date_col: str = 'Date') -> float:
    """
    Fits a Gumbel distribution to annual maximum discharge values and calculates
    the discharge corresponding to a given return period.

    :param hydro_df: A DataFrame containing discharge data with date and value columns.
    :param RP: The return period in years.
    :param date_col: Name of the column containing datetime information (default: 'ValidTime').
    :param value_col: Name of the column containing discharge values (default: 'Value').
    :return: The discharge value corresponding to the return period.
    """
    loc, scale = Q_Gumbel_fit (hydro_df, value_col, timescale='Year', date_col=date_col)
    return gumbel_r.ppf(1 - 1 / RP, loc, scale)


def Q_GEV_fit_RP(hydro_df, value_col, RP):
    """
    Fits a GEV distribution to a time series of discharge values and calculates
    the discharge values corresponding to specified return periods.

    :param series: A pandas Series containing discharge values.
    :param return_periods: A list of return periods in years.
    :return: A dictionary with return periods as keys and corresponding discharge values.
    """

    shape,loc, scale = Q_GEV_fit(hydro_df, value_col, timescale='Year')
    discharge_value = genextreme.ppf(1-1/RP, shape, loc=loc, scale=scale)
    return discharge_value


def Q_GEV_fit_RP_EVA(series: pd.Series, return_periods: list = [2, 5, 10, 25, 50, 100]) -> dict:
    """
    Fits a GEV distribution to a time series of discharge values and calculates
    the discharge values corresponding to specified return periods.

    :param series: A pandas Series containing discharge values.
    :param return_periods: A list of return periods in years.
    :return: A dictionary with return periods as keys and corresponding discharge values.
    """
    model = EVA(series)
    model.get_extremes(method='BM', block_size="365D")
    gev_fit = model.fit_model()
    model.plot_diagnostic(alpha=0.95)
    summary = model.get_summary(
        return_period=return_periods,
        alpha=0.95,  # confidence interval
        n_samples=1000,
    )
    return summary

def plot_diagnostics(series, shape_gev, loc_gev, scale_gev, loc_gumbel, scale_gumbel, rp_values):
    """
    Generate diagnostic and comparison plots for GEV and Gumbel fits.

    Parameters:
        series (pd.Series): Time series of discharge values.
        shape_gev, loc_gev, scale_gev: GEV parameters.
        loc_gumbel, scale_gumbel: Gumbel parameters.
        rp_values (list): List of return periods.
    """
    # Sort data for plotting purposes
    sorted_data = np.sort(series)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Theoretical values
    gev_theoretical = genextreme.ppf(ecdf, shape_gev, loc=loc_gev, scale=scale_gev)
    gumbel_theoretical = gumbel_r.ppf(ecdf, loc=loc_gumbel, scale=scale_gumbel)

    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Q-Q plot
    axes[0, 0].scatter(gev_theoretical, sorted_data, label='GEV', color='b')
    axes[0, 0].scatter(gumbel_theoretical, sorted_data, label='Gumbel', color='r', alpha=0.6)
    axes[0, 0].plot([min(sorted_data), max(sorted_data)], [min(sorted_data), max(sorted_data)], '--k')
    axes[0, 0].set_title("Q-Q Plot")
    axes[0, 0].set_xlabel("Theoretical")
    axes[0, 0].set_ylabel("Observed")
    axes[0, 0].legend()

    # P-P plot
    axes[0, 1].scatter(ecdf, genextreme.cdf(sorted_data, shape_gev, loc=loc_gev, scale=scale_gev), label='GEV', color='b')
    axes[0, 1].scatter(ecdf, gumbel_r.cdf(sorted_data, loc=loc_gumbel, scale=scale_gumbel), label='Gumbel', color='r', alpha=0.6)
    axes[0, 1].plot([0, 1], [0, 1], '--k')
    axes[0, 1].set_title("P-P Plot")
    axes[0, 1].set_xlabel("Empirical CDF")
    axes[0, 1].set_ylabel("Theoretical CDF")
    axes[0, 1].legend()

    # Return Period Plot
    gev_return = genextreme.ppf(1 - 1 / np.array(rp_values), shape_gev, loc=loc_gev, scale=scale_gev)
    gumbel_return = gumbel_r.ppf(1 - 1 / np.array(rp_values), loc=loc_gumbel, scale=scale_gumbel)
    axes[1, 0].scatter(rp_values, series.max(), color='k', label="Observed Data")
    axes[1, 0].plot(rp_values, gev_return, '-b', label="GEV Fit")
    axes[1, 0].plot(rp_values, gumbel_return, '-r', label="Gumbel Fit")
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_title("Return Period Plot")
    axes[1, 0].set_xlabel("Return Period (years)")
    axes[1, 0].set_ylabel("Discharge")
    axes[1, 0].legend()

    # Probability Density Function
    x = np.linspace(min(series), max(series), 1000)
    gev_pdf = genextreme.pdf(x, shape_gev, loc=loc_gev, scale=scale_gev)
    gumbel_pdf = gumbel_r.pdf(x, loc=loc_gumbel, scale=scale_gumbel)
    axes[1, 1].hist(series, bins=20, density=True, alpha=0.4, color='g', edgecolor='k')
    axes[1, 1].plot(x, gev_pdf, '-b', label="GEV PDF")
    axes[1, 1].plot(x, gumbel_pdf, '-r', label="Gumbel PDF")
    axes[1, 1].set_title("Probability Density Function")
    axes[1, 1].set_xlabel("Discharge")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
def flood_frequency_curve (hydro_df, value_col):
    annual_max = calculate_max_discharge (hydro_df, value_col, timescale='Year')
    n = len(annual_max)
    return_periods = (n + 1) / np.arange(1, n + 1)  # Return period = (N + 1) / Rank

    # Step 3: Fit distributions
    # Gumbel Distribution
    gumbel_params = gumbel_r.fit(annual_max)
    gumbel_rv = gumbel_r(*gumbel_params)

    # GEV Distribution
    gev_params = genextreme.fit(annual_max)
    gev_rv = genextreme(*gev_params)

    # Step 4: Calculate theoretical frequencies
    return_periods_theoretical = np.linspace(1.01, 20, 5)  # Return periods for smooth curves
    probabilities_theoretical = 1 - 1 / return_periods_theoretical

    # Quantiles for Gumbel and GEV
    gumbel_quantiles = gumbel_rv.ppf(probabilities_theoretical)
    gev_quantiles = gev_rv.ppf(probabilities_theoretical)
    gev_quantiles 
    # Step 5: Plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(return_periods, annual_max[::-1], 'o', label="GloFAS data", color='black')  # Observed data
    plt.plot(return_periods_theoretical, gumbel_quantiles, label="Gumbel Distribution", color='blue')
    #plt.plot(return_periods_theoretical, gev_quantiles, label="GEV Distribution by Scipy", color='red')

    # Formatting the plot
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Return Period (years)")
    plt.ylabel("Flood Magnitude")
    plt.title("Flood Frequency Curve")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    station = 'Bamako'
    leadtime = 168
    return_periods = [1, 1.5, 2, 5, 10, 25, 50, 100]  
    Q_Bamako_csv = f"{cfg.stationsDir}/GloFAS_Q/timeseries/discharge_timeseries_{station}_{leadtime}.csv"

    df = pd.read_csv(
        Q_Bamako_csv,
        index_col=0,  # setting ValidTime as index column
        parse_dates=True,  # parsing dates in ValidTime
    )

    df['percentile_40.0'] = df['percentile_40.0'].interpolate(method='time')
    flood_frequency_curve(df, value_col='percentile_40.0')
    series = csv_to_cleaned_series(Q_Bamako_csv)
    
    # GEV modeling
    model = EVA(series)
    model.get_extremes(method='BM', block_size="365D")
    gev_fit = model.fit_model()
    model.plot_diagnostic(alpha=0.95)
    summary = model.get_summary(
        return_period=return_periods,
        alpha=0.95,  # confidence interval
        n_samples=1000,
    )
    print(summary)

    # Percentile calculations
    percentile95_GEV = Q_GEV_fit_percentile(df, 95, value_col='percentile_40.0')
    percentile95_Gumbel = Q_Gumbel_fit_percentile(df, 95,value_col='percentile_40.0')


    print(f"95th Percentile (GEV): {percentile95_GEV}")
    print(f"95th Percentile (Gumbel): {percentile95_Gumbel}")

    # shape_gev, loc_gev, scale_gev = Q_GEV_fit (df, value_col='percentile_40.0', timescale='Year')
    # loc_gumbel, scale_gumbel = Q_Gumbel_fit(df, value_col='percentile_40.0')
    # plot_diagnostics(series, shape_gev, loc_gev, scale_gev, loc_gumbel, scale_gumbel, return_periods)