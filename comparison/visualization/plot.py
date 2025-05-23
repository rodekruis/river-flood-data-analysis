import cmcrameri.cm as cmc #change colormaps
import pandas as pd
from datetime import datetime, timedelta
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
from GloFAS.GloFAS_prep.text_formatter import capitalize
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from comparison.collect_for_administrative_unit import collect_performance_measures_over_admin, collect_performance_measures_over_station
import GloFAS.GloFAS_prep.configuration as cfg
from comparison.observation.HydroImpact import transform_hydro
from comparison.observation.thresholds import Q_Gumbel_fit_RP
class Visualizer: 
    def __init__(self, DataDir, vector_adminMap, leadtimes, RPsyr, percentiles):
        self.models = ['GloFAS', 'GoogleFloodHub', 'PTM'] # is PTM best way to refer to the current trigger model in the EAP? 
        self.colors = ['cornflowerblue', 'darkorange','darkgreen'] # 0 observation, 1 glofas, 2 gfh
        self.markersize = 12
        self.admin_colors = [
            "#a6cee3", "gold", "cornflowerblue", "#ff7f00", "#4daf4a", "#e41a1c", "#984ea3", 
            "#a65628", "#f781bf", "#999999", 
            "#009E73", "#0072B2", "olive"
            ]#['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', ] # adjust pls if you want 
        self.linestyles = [':', '-','-.' ]
        self.markerstyles =['s','o', 'v']
        self.comparisonTypes = ['Observation']#, 'Impact']
        self.percentiles = percentiles
        self.RPsyr = RPsyr
        self.leadtimes = leadtimes
        self.DataDir=DataDir
        self.gdf_shape=checkVectorFormat(vector_adminMap, shapeType='polygon')
        self.cmap_r = 'RdYlBu_r' # 'cmc.batlow_r'  # Reversed for FAR
        self.cmap = 'RdYlBu' # 'cmc.batlow'    # Default for POD
        self.POD_threshold = 0.6 # to colour everything below it red
        self.FAR_threshold = 0.3 # to colour everything above it red 
        self.dpi_quality = 500
    def map_performance(self, scores_by_commune_gdf, RPyr, leadtime, comparisonType, model):
        """
        Visualize performance metrics (POD, FAR, CSI, POFD, Accuracy, Precision) on separate maps.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        plt.suptitle(f'Performance Metrics for Flood Prediction by {model} (RP{RPyr:.1f} Year, Lead Time: {leadtime/24:.0f} Days), compared by {comparisonType} data')

        # Titles and metrics to display
        titles = [
            'POD (Probability of Detection)', 'FAR (False Alarm Ratio)', 'CSI (Critical Success Index)',
            'POFD (Probability of False Detection)', 'Accuracy', 'Precision'
            ]
        metrics = ['pod', 'far', 'csi', 'pofd', 'accuracy', 'precision']


        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            # Plot each metric
            self.gdf_shape.plot(ax=ax, color='lightgrey', alpha=0.5)
            if title in ['POD (Probability of Detection)', 'CSI (Critical Success Index)', 'Accuracy', 'Precision']:
                scores_by_commune_gdf.plot(column=metric, cmap=self.cmap_r, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, vmin=0, vmax=1)
            else:
                scores_by_commune_gdf.plot(column=metric, cmap=self.cmap, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, vmin=0, vmax=1)
            
            ax.set_title(title)
            ax.set_axis_off()

        plt.tight_layout()
        filePath = f'{self.DataDir}/GloFAS/{comparisonType}/results/performance_metrics_RP{RPyr:.1f}_yr_leadtime{leadtime:.0f}hrs.png'
        plt.savefig(filePath)
        plt.show()
    def map_pod_far (self, scores_by_commune_gdf, RPyr, leadtime, comparisonType, model):
        """
        Visualize selected performance metrics (POD, FAR) on a two-panel map.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        plt.suptitle(f'Performance Metrics for Flood Prediction by {model} (RP{RPyr:.1f} Year, Lead Time: {leadtime/24:.0f} days), compared by {comparisonType} data')

        # Titles and metrics to display
        titles = ['POD (Probability of Detection)', 'FAR (False Alarm Ratio)']
        if model == 'GloFAS':
            metrics = ['pod', 'far']
        elif model == 'GoogleFloodHub': 
            metrics = ['POD', 'FAR']
        # Define color maps
        for ax, metric, title, cmap in zip(axes, metrics, titles, [self.cmap, self.cmap_r]):
            # Plot each metric
            self.gdf_shape.plot(ax=ax, color='lightgrey', alpha=0.5)
            scores_by_commune_gdf.plot(column=metric, cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, vmin=0, vmax=1)
            
            ax.set_title(title)
            ax.set_axis_off()

        plt.tight_layout()
        filePath = f'{self.DataDir}/{model}/{comparisonType}/results/performance_metrics_RP{RPyr:.1f}_yr_leadtime{leadtime:.0f}hrs_POD_FAR.png'
        plt.savefig(filePath)
        plt.show()


    def performance_metrics(self, 
                            spatial_units, 
                            spatial_unit_type='StationNames',
                            x_axis ='leadtime', 
                            standard_value=168, 
                            threshold_type='return_periods'):
        """
        Plot performance metrics for either lead time or return period.
        
        Parameters:
        - spatial_units: list of strings: describing station names or admin units
        - spatial_unit_type: str: either : 'StationNames' or 'AdminUnits'
        - x_axis: 'leadtime' or 'return_period' or 'percentiles'
        - standard_value: Standard leadtime (hours) or return period (years) for slicing, needs to be of the other type than the x-axis, because this will be standard value 
        - threshold_type: Threshold type for return periods ('return_periods' by default)
        """

        data_all_admin = []
        if spatial_unit_type =='StationNames':
            for spatial_unit in spatial_units:
                data = collect_performance_measures_over_station(
                                                    spatial_unit, 
                                                    self.DataDir, 
                                                    self.leadtimes, 
                                                    self.RPsyr, 
                                                    self.percentiles)
                data_all_admin.append((spatial_unit, data))
        elif spatial_unit_type =='AdminUnits': 
            for spatial_unit in spatial_units:
                data = collect_performance_measures_over_admin(
                                                    spatial_unit, 
                                                    self.DataDir, 
                                                    self.leadtimes, 
                                                    self.RPsyr, 
                                                    self.percentiles)
                data_all_admin.append((spatial_unit, data))
        else: 
            raise ValueError (f"choose spatial_unit_type of 'AdminUnits' or 'StationNames', not {spatial_unit_type}")


        # Determine axis values based on mode
        if x_axis == 'leadtime':
            figsize = (7, 5)
            x_values = self.leadtimes
            x_label = 'Leadtime (hours)'
            x_lim = [min(x_values)- 3, max(x_values)+3]
            if threshold_type=='return_periods':
                standard_idx = self.RPsyr.index(standard_value)
                secondary_text = f'Return period={standard_value:.1f} years'
            elif threshold_type=='percentiles': 
                standard_idx = self.percentiles.index(standard_value)
                secondary_text = f'Percentile={standard_value:.0f}%'
            else: 
                raise ValueError (f"choose threshold_type of type 'return_periods' or 'percentiles', not {threshold_type}")
        elif x_axis =='percentile': 
            figsize=(6, 6) # making a higher but more narrow plot
            x_values = self.percentiles 
            x_label = 'Percentile (%)'
            standard_idx = self.leadtimes.index(standard_value)
            x_lim = [min(x_values) - 0.1, max(x_values) + 0.1]
            secondary_text = f'Leadtime={standard_value:.0f} hours ({standard_value / 24:.0f} days)'
        elif x_axis == 'return_period':
            figsize=(7, 5)
            x_values = self.RPsyr
            x_label = 'Return Period (years)'
            x_lim = [min(x_values) - 0.2, max(x_values) + 0.2]
            standard_idx = self.leadtimes.index(standard_value)
            secondary_text = f'Leadtime={standard_value:.0f} hours ({standard_value / 24:.0f} days)'
        else: 
            raise ValueError (f"choose x_axis 'return_period', 'percentile' or 'leadtime', not {x_axis}")
        
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f'Performance metrics for stations over {x_axis}',fontsize=12) # {"lead time" if x_axis == "leadtime" el "return period"}',
        # Plot POD
        ax = axs[0]
        for spatial_unit, color in zip(spatial_units, self.admin_colors):
            for model, marker, linestyle in zip(self.models, self.markerstyles, self.linestyles):
                if x_axis == 'leadtime':
                    y_values = data_all_admin[spatial_units.index(spatial_unit)][1]['POD'][model]['Observation'][threshold_type][standard_idx, :]
                else:
                    y_values = data_all_admin[spatial_units.index(spatial_unit)][1]['POD'][model]['Observation'][threshold_type][:, standard_idx]
                ax.plot(x_values, y_values, c=color, linestyle=linestyle, marker=marker, markersize=self.markersize , markerfacecolor=color, markeredgecolor = 'white', alpha=0.75)
        #ax.set_xlabel(x_label)
        ax.set_ylabel('POD')
        ax.set_xlim(x_lim)
        ax.set_ylim([-0.05, 1.05])
        ax.fill_between(x_lim, ax.get_ylim()[0], self.POD_threshold, color='maroon', alpha=0.15)
        ax.text(x_lim[0] + ((x_lim[1]-x_lim[0])/100), 1.10, secondary_text)

        # Plot FAR
        ax = axs[1]
        for spatial_unit, color in zip(spatial_units, self.admin_colors):
            for model, marker, linestyle in zip(self.models, self.markerstyles, self.linestyles):
                if x_axis == 'leadtime':
                    y_values = data_all_admin[spatial_units.index(spatial_unit)][1]['FAR'][model]['Observation'][threshold_type][standard_idx, :]
                else:
                    y_values = data_all_admin[spatial_units.index(spatial_unit)][1]['FAR'][model]['Observation'][threshold_type][:, standard_idx]
                ax.plot(x_values, y_values, 
                        c=color, 
                        linestyle=linestyle, 
                        marker=marker, 
                        markersize=self.markersize, 
                        markerfacecolor=color, 
                        markeredgecolor ='white', 
                        alpha=0.75) #, markerfacecolor='none')
        ax.set_xlabel(x_label)
        ax.set_ylabel('FAR')
        ax.set_xlim(x_lim)
        ax.set_ylim([-0.05, 1.05])
        ax.fill_between(x_lim, ax.get_ylim()[1], self.FAR_threshold, color='maroon', alpha=0.15)
        #ax.text(x_lim[0] + ((x_lim[1]-x_lim[0])/100), 1.10, secondary_text)

        # Custom Legend
        
        custom_lines = [
            Line2D([0], [0], color='black', marker=self.markerstyles[0], linestyle=self.linestyles[0], label='GloFAS'),
            Line2D([0], [0], color='black', marker=self.markerstyles[1], linestyle=self.linestyles[1], label='Google FloodHub'),
            Line2D([0], [0], color='black', marker=self.markerstyles[2], linestyle=self.linestyles[2], label='PTM'),
        ]
        color_lines = [Line2D([0], [0], color=color, lw=2, label=unit, alpha=0.75) for color, unit in zip(self.admin_colors, spatial_units)]
        fig.legend(handles=custom_lines + color_lines,
                loc='lower center', ncol=4, fontsize='small', frameon=False)

        plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Adjusted spacing for legend
        file_path = f'{self.DataDir}/comparison/results/performance_metrics_over{x_axis}_all_admin_otherv{standard_value}.png'
        plt.savefig(file_path, dpi=self.dpi_quality)
        plt.show()
        plt.close('all')
        


    def plot_flood_and_impact_events(self, 
                                    df_glofas, 
                                    df_gfh, 
                                    df_impact, 
                                    df_obs, 
                                    stationname, 
                                    admin_unit, 
                                    leadtime, 
                                    return_period, 
                                    start_time, 
                                    end_time, 
                                    threshold_glofas, 
                                    threshold_gfh, 
                                    threshold_obs,
                                    second_return_period=None, 
                                    second_threshold_glofas=None,
                                    second_threshold_gfh=None,
                                    second_threshold_obs=None):
        """
        Plots observed discharge, predicted discharge, and flood impact events for a given administrative unit.
        
        Parameters:
        - df_glofas (pd.DataFrame): Predicted discharge with 'Date' as index and 'Value' as discharge.
        - df_impact (pd.DataFrame): Impact events with columns 'admin_unit', 'Start Date', and 'End Date'.
        - df_obs (pd.DataFrame): Observed discharge with 'Date' as index and 'Value' as discharge.
        - admin_unit (str): Administrative unit to filter impact events.
        - start_time (str): Start time for the plot (format 'YYYY-MM-DD').
        - end_time (str): End time for the plot (format 'YYYY-MM-DD').
        - threshold (float): Discharge threshold (e.g., 5-year return period).
        """
        # Filter impact events for the specific admin unit
        df_impact_bamako_events = df_impact[df_impact['ADM2'] == capitalize(admin_unit)]

        # Convert start and end times to datetime for filtering
        start_time = pd.to_datetime(start_time, format="%Y-%m-%d")
        end_time = pd.to_datetime(end_time, format="%Y-%m-%d")

        # Filter observed and predicted discharge for the time range
        df_obs = df_obs.loc[(df_obs.index >= start_time) & (df_obs.index <= end_time)]
        df_glofas = df_glofas.loc[(df_glofas.index >= start_time) & (df_glofas.index <= end_time)]
        df_gfh = df_gfh.loc[(df_gfh.index >= start_time) & (df_gfh.index <= end_time)]

        # Plot the data
        plt.figure(figsize=(12, 7))

        # Plot observed discharge (solid line)
        plt.plot(df_obs.index, df_obs['Value'], label='Observed discharge', color=self.colors[0], linestyle='-', linewidth=1.5)

        # Plot predicted discharge for GloFAS and GFH (solid line)
        plt.plot(df_glofas.index, df_glofas['percentile_40.0'], label=f'GloFAS (leadtime {leadtime}h)', color=self.colors[1], linestyle='-', linewidth=1.5)
        plt.plot(df_gfh.index, df_gfh[f'{leadtime/24:.0f} days'], label=f'GFH (leadtime {leadtime}h)', color=self.colors[2], linestyle='-', linewidth=1.5)

        # Add thresholds for return periods
        plt.axhline(threshold_obs, color=self.colors[0], linestyle='--', linewidth=2, label=f'{return_period}-yr RP (Obs)')
        plt.axhline(threshold_glofas, color=self.colors[1], linestyle='--', linewidth=2, label=f'{return_period}-yr RP (GloFAS)')
        plt.axhline(threshold_gfh, color=self.colors[2], linestyle='--', linewidth=2, label=f'{return_period}-yr RP (GFH)')

        if second_return_period is not None:
            plt.axhline(second_threshold_obs, color=self.colors[0], linestyle=':', linewidth=2, label=f'{second_return_period}-yr RP (Obs)')
            plt.axhline(second_threshold_glofas, color=self.colors[1], linestyle=':', linewidth=2, label=f'{second_return_period}-yr RP (GloFAS)')
            plt.axhline(second_threshold_gfh, color=self.colors[2], linestyle=':', linewidth=2, label=f'{second_return_period}-yr RP (GFH)')

        # Add impact events as translucent red blocks
        for _, row in df_impact_bamako_events.iterrows():
            start = pd.to_datetime(row['Start Date'])
            end = pd.to_datetime(row['End Date'])
            if start >= start_time and end <= end_time:
                plt.axvspan(start, end, color='maroon', alpha=0.7, label='Impact event' if 'Impact event' not in plt.gca().get_legend_handles_labels()[1] else "")

        # Customize plot
        plt.title(f'Discharge at {stationname} and impact events in {admin_unit}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Discharge (m³/s)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Simplified legend with horizontal spreading

        custom_lines = [
            Line2D([0], [0], color=self.colors[0], label='GloFAS discharge (168 hours leadtime)'),
            Line2D([0], [0], color=self.colors[1],  label='Google FloodHub discharge (168 hours leadtime)'),
            Line2D([0], [0], color=self.colors[2],  label='Observed discharge'),
            Line2D([0], [0],  color = 'black', linestyle='--', label=f'{return_period}-yr RP'),
            Line2D([0], [0],  color ='black', linestyle=':', label=f'{second_return_period}-yr RP'),
            Line2D([0], [0], color='maroon', alpha=0.7, linestyle='-', linewidth=8, label='Impact event') ,
            ]

        plt.legend(handles=custom_lines,
                 loc='lower center', 
                 ncol=2, 
                 fontsize='medium', 
                 frameon=False, 
                 bbox_to_anchor=(0.5, -0.24))
        plt.tight_layout() 

        file_path = f'{self.DataDir}/comparison/results/discharge_in{stationname}_timeseries_RP{return_period}_leadtime{leadtime}.png'
        plt.savefig(file_path)
        # Show plot
        plt.show()

if __name__ =='__main__': 
    vis = Visualizer(cfg.DataDir, cfg.admPath, cfg.leadtimes, cfg.RPsyr, cfg.percentiles)
    # comparisonType = 'Impact'
    # model = 'GoogleFloodHub'
    # for RPyr in cfg.RPsyr: 
    #     for leadtime in cfg.leadtimes: 
    #         if RPyr == int (RPyr): 
    #             RPyr = int(RPyr)
    #         else:
    #             RPyr = float(RPyr)
    #         scores_path = f"{cfg.DataDir}/{model}/{comparisonType}/GFH_vs_IMPACT_{leadtime:.0f}lt_{RPyr}rp.geojson"
    #         scores_by_commune_gdf = checkVectorFormat(scores_path)
    #         vis.map_pod_far (scores_by_commune_gdf, RPyr, leadtime, comparisonType, model)

    BasinName = 'Niger'
    StationName = 'Bamako'
    CorrespondingAdminUnit = 'Bamako'
    
    leadtime = 168
    return_period = 5 # year 
    second_return_period = 1.5  
    df_obs = transform_hydro(f"{cfg.DataDir}/DNHMali_2019/Q_stations/{BasinName}_{StationName}.csv",cfg.startYear, cfg.endYear)
    df_glofas = pd.read_csv (f"{cfg.stationsDir}/GloFAS_Q/timeseries/discharge_timeseries_{StationName}_{leadtime}.csv")
    df_gfh = pd.read_csv (f"{cfg.DataDir}/GoogleFloodHub/timeseries/Bamako_streamflow.csv")
    df_glofas3 = pd.read_csv(f"{cfg.DataDir}/GloFAS31_threshold/discharge_reanalysis_-8.05_12.55.csv")

    df_glofas3 ['ValidTime'] = pd.to_datetime(df_glofas3["ValidTime"], format="%Y-%m-%d")
    df_gfh ['ValidTime'] = pd.to_datetime(df_gfh["issue_time"], format="%Y-%m-%d") + timedelta(days=leadtime/24)
    df_glofas['ValidTime'] = pd.to_datetime(df_glofas["ValidTime"], format="%Y-%m-%d")

    # Check for any invalid dates
    if df_glofas['ValidTime'].isnull().any():
        print("Warning: Some dates in 'date_col' ofglofas could not be parsed and are set to NaT.")
    if df_gfh['ValidTime'].isnull().any():
        print("Warning: Some dates in 'date_col' of gfh could not be parsed and are set to NaT.")
    if df_glofas3['ValidTime'].isnull().any():
        print ("Warning: Some dates in date_col could not be parsed and are set to NaT")
    # Set the index to the datetime column
    df_glofas.set_index('ValidTime', inplace=True)
    df_glofas.sort_index(inplace=True)
    df_glofas3.set_index('ValidTime', inplace=True)
    df_glofas3.sort_index(inplace=True)

    df_gfh.set_index('ValidTime', inplace=True)
    df_gfh.sort_index(inplace=True)

    df_impact = pd.read_csv (cfg.impact_csvPath, delimiter=';', header=0)
    RP_gfh = Q_Gumbel_fit_RP (df_gfh, return_period, f'{leadtime/24:.0f} days')
    return_period=5
    print (df_glofas3.columns)


    RP_glofas = Q_Gumbel_fit_RP(df_glofas, return_period, 'percentile_40.0')
    RP_glofas3 = Q_Gumbel_fit_RP(df_glofas3, return_period, 'Discharge')
    RP_obs = Q_Gumbel_fit_RP (df_obs, return_period, 'Value')
    print (f'glofas: {RP_glofas}, glofas3: {RP_glofas3}, obs: {RP_obs}')
    
    second_RP_gfh = Q_Gumbel_fit_RP (df_gfh, second_return_period, f'{leadtime/24:.0f} days')
    second_RP_glofas = Q_Gumbel_fit_RP(df_glofas, second_return_period, 'percentile_40.0')
    second_RP_obs = Q_Gumbel_fit_RP (df_obs, second_return_period, 'Value')

    vis.plot_flood_and_impact_events(df_glofas,
                                        df_gfh, 
                                        df_impact, 
                                        df_obs, 
                                        StationName,
                                        CorrespondingAdminUnit, 
                                        leadtime,
                                        return_period,
                                        '2016-07-01', 
                                        '2019-01-01', 
                                        RP_glofas, 
                                        RP_gfh,
                                        RP_obs, 
                                        second_return_period,
                                        second_RP_glofas,
                                        second_RP_gfh, 
                                        second_RP_obs)
    # comparisonTypes = ['Impact','Observation']
    # models = ['GloFAS','GoogleFloodHub', 'PTM']
    # for model in models: 
    #     for comparisonType in comparisonTypes:
    #         for leadtime in cfg.leadtimes: 
    #             for RPyr in cfg.RPsyr:
    #                 try:
    #                     scores_path = f"{cfg.DataDir}/{model}/{comparisonType}/scores_byCommuneRP{RPyr:.1f}_yr_leadtime{leadtime:.0f}.gpkg"
    #                     scores_by_commune_gdf = checkVectorFormat(scores_path)
    #                     vis.map_pod_far(scores_by_commune_gdf, RPyr, leadtime, comparisonType, f'{model}')
    #                 except:
    #                     print (f'No path for leadtime: {leadtime}, RP: {RPyr}, {comparisonType}, for model: {model}') 
    #                     continue
    # removed gao and ansongo from stations 

    station_names = ['BAFING MAKANA',  'BANANKORO','MOPTI','ANSONGO', 'DIRE', 'SOFARA', 'DOUNA', 'BOUGOUNI', 'PANKOUROU','DIBIYA', 'KAYES', 'KOULIKORO', 'BAMAKO']
    for leadtime in cfg.leadtimes:
        vis.performance_metrics(station_names, spatial_unit_type='StationNames', x_axis='percentile', standard_value=leadtime, threshold_type='percentiles')
        vis.performance_metrics(station_names, spatial_unit_type='StationNames', x_axis='return_period', standard_value=leadtime, threshold_type='return_periods')
    for RPyr in cfg.RPsyr:
        vis.performance_metrics (station_names, spatial_unit_type='StationNames', x_axis='leadtime', standard_value=RPyr)
    for percentile in cfg.percentiles: 
        vis.performance_metrics (station_names, spatial_unit_type='StationNames', x_axis='leadtime', standard_value=percentile, threshold_type='percentiles')

            

# to do : 
## add lines between points 
## general layouts 