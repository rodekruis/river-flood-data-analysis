# NEXT, plotting!
# exclude all datasets where the 5-yr RP is below 100
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates

STATIONS = {
    "Mopti": {
        "gfh": "?",
        "glofas_v4": "../../GloFAS/data/timeseries/discharge_timeseries_Mopti_168.csv",
        "obs": "../../comparison/observation/data/DNHMali_2019/Q_stations/Niger_Mopti.csv",
    },
    "Ansongo": {
        "gfh": "?",
        "glofas_v4": "../../GloFAS/data/timeseries/discharge_timeseries_Ansongo _168.csv",
        "obs": "../../comparison/observation/data/DNHMali_2019/Q_stations/Niger_Ansongo.csv",
    }, 
    "Sofara": {
        "gfh": "?",
        "glofas_v4": "../../GloFAS/data/timeseries/discharge_timeseries_Sofara_168.csv",
        "obs": "../../comparison/observation/data/DNHMali_2019/Q_stations/Niger_Sofara.csv",
    },
}



def get_dataset_with_ID(dict_datasets, ID):
    if f"ds_{ID}" in dict_datasets:
        return dict_datasets[f"ds_{ID}"]
    elif f"ds_reforecast_{ID}" in dict_datasets:
        return dict_datasets[f"ds_reforecast_{ID}"]
    return None


def load_google_flood_hub(place_name, dict_datasets):

    station = STATIONS[place_name]

    ds = get_dataset_with_ID(dict_datasets, station["gfh_id"])

    df = ds["streamflow"].to_dataframe().drop(columns="gauge_id")
    df = df.unstack(level="lead_time")
    df.columns = df.columns.droplevel()
    df.index = pd.to_datetime(df.index)

    df_4d = pd.DataFrame(df["4 days"])
    df_4d.columns = ["Q"]

    df_4d["actual_date"] = df_4d.index + pd.Timedelta(days=4)
    df_4d.set_index("actual_date", inplace=True)

    return df_4d


def load_glofas_v4(place_name):

    station = STATIONS[place_name]

    df = pd.read_csv(station["glofas_v4"])

    df = df[["ValidTime", "percentile_40.0"]]
    df.columns = ["actual_date", "Q"]

    df["actual_date"] = pd.to_datetime(df["actual_date"])
    df.set_index("actual_date", inplace=True)
    df.sort_index(inplace=True)

    return df



def load_observations(place_name):

    station = STATIONS[place_name]

    df_obs = pd.read_csv(station["obs"])

    df = df_obs.melt(
        id_vars=["Date"],
        var_name="Year",
        value_name="Value",
    )

    def parse_date(date_str, year):
        try:
            return pd.to_datetime(
                f"{year} {date_str}",
                format="%Y %d/%m %H:%M",
            )
        except ValueError:
            return None

    df["actual_date"] = df.apply(
        lambda row: parse_date(row["Date"], row["Year"]),
        axis=1,
    )

    df.dropna(subset=["actual_date"], inplace=True)

    df["actual_date"] = df["actual_date"].dt.normalize()

    df.set_index("actual_date", inplace=True)

    df = df[["Value"]]
    df.rename(columns={"Value": "Q"}, inplace=True)

    return df


def load_hydrograph_data(place_name):

    return {
        "observations": load_observations(place_name),
        "GloFAS v4.0": load_glofas_v4(place_name),
        #"Google Flood Hub": load_google_flood_hub(place_name),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_hydrograph(
    ax,
    place_name,
    start_date="2018-03-01",
    end_date="2018-12-31",
):

    d_dfs = load_hydrograph_data(place_name)

    colours = [ "black", "magenta", 'blue']
    linetypes = ["-","--","-."]
    markers = [ "D", "^","o"]
    marker_ints = [120, 135, 150, 165]

    for (name, df), colour, ls, marker, mi in zip(
        d_dfs.items(),
        colours,
        linetypes,
        markers,
        marker_ints,
    ):

        df = df.loc[start_date:end_date]

        ax.plot(
            df.index,
            df["Q"],
            label=name,
            color=colour,
            linestyle=ls,
            marker=marker,
            markersize=3,
            markevery=mi,
            lw=1.2,
        )

    ax.set_title(place_name)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))


def plot_hydrographs(place_names=["Mopti", "Ansongo"],
    start_date="2018-03-01",
    end_date="2018-12-31",
):

    fs = 12

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4 * 1.618, 7),
        sharex=True,
    )

    plot_hydrograph(
        axes[0],
        place_names[0],
        start_date,
        end_date,
    )

    plot_hydrograph(
        axes[1],
        place_names[1],
        start_date,
        end_date,
    )

    axes[0].legend(fontsize=fs, frameon=False)

    for ax in axes:
        ax.set_ylabel(
            r"discharge [$\mathrm{m}^3/\mathrm{s}$]",
            fontsize=fs,
        )
        ax.tick_params(axis="both", labelsize=fs)

    plt.tight_layout()
    plt.show()


#plot_hydrographs(start_date="2018-03-01", end_date="2018-12-31")
plot_hydrographs(place_names=["Sofara", "Ansongo"], start_date="2016-03-01", end_date="2018-12-31")