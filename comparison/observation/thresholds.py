import pandas as pd
import GloFAS.GloFAS_prep.configuration as cfg
import pyextremes as pe
from pyextremes import EVA
from scipy.stats import genextreme

def csv_to_cleaned_series (csv): 
    df = pd.read_csv(
        Q_Bamako_csv,
        index_col=0, # setting ValidTime as index column
        parse_dates=True, # parsing dates in ValidTime
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
def Q_Gumbel_fit_RP (hydro_df, RP): 

    # Extract the annual maximum discharge values
    hydro_df['Year'] = hydro_df['Date'].dt.year
    annual_max_discharge = hydro_df.groupby('Year')['Value'].max()

    # Fit a Gumbel distribution to the annual maximum discharge values
    loc, scale = stats.gumbel_r.fit(annual_max_discharge)
    # Calculate the discharge value corresponding to the return period
    discharge_value = stats.gumbel_r.ppf(1 - 1/RP, loc, scale)
    return discharge_value

def Q_Gumbel_fit_percentile (hydro_df, percentile): 
    # now for 
    return discharge_value

def Q_GEV_fit_RP (hydro_df, RP):  
    return discharge_value

def Q_GEV_fit_percentile (hydro_df, percentile): 
    """
    Fits a GEV distribution to the daily discharge values and calculates the discharge 
    corresponding to a given percentile.

    Parameters:
        hydro_df (pd.DataFrame): A dataframe with at least 'Date' and 'Value' columns.
            'Date' should be a datetime object and 'Value' is the discharge value.
        percentile (float): Percentile for which to compute the discharge value (between 0 and 100).

    Returns:
        float: The discharge value corresponding to the given percentile.
    """
    # Ensure 'Date' column is a datetime object
    hydro_df['Date'] = pd.to_datetime(hydro_df['Date'])

    # Extract daily discharge values
    daily_discharge = hydro_df['Value']
    #  Remove non-finite values
    daily_discharge_cleaned = daily_discharge[np.isfinite(daily_discharge)]

    # Check if there are still issues
    if daily_discharge_cleaned.empty:
        raise ValueError("All data was non-finite after cleaning. Please check your dataset.")

    # Fit a GEV distribution
    shape, loc, scale = genextreme.fit(daily_discharge_cleaned)


    # Calculate the discharge value corresponding to the specified percentile
    discharge_value = genextreme.ppf(percentile / 100, shape, loc=loc, scale=scale)

    return discharge_value

if __name__ == '__main__': 
    station = 'Bamako'
    leadtime = 168 
    Q_Bamako_csv = f"{cfg.stationsDir}/GloFAS_Q/timeseries/discharge_timeseries_{station}_{leadtime}.csv"
    series = csv_to_cleaned_series(Q_Bamako_csv)
    model = EVA(series)
    model.get_extremes(
        method='BM',  # Block Maxima method
        block_size="365D",  # One year per block
        )

    #model.plot_extremes()
    gev_fit = model.fit_model()
    model.plot_diagnostic(alpha=0.95)
    summary = model.get_summary(
        return_period=[1, 1.5, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
        alpha=0.95, # confidence interval
        n_samples=1000,
        )
    print (summary)

