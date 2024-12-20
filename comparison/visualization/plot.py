import cmcrameri.cm as cmc #change colormaps
import pandas as pd
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
from GloFAS.GloFAS_prep.text_formatter import capitalize
import matplotlib.pyplot as plt
from comparison.collect_for_administrative_unit import collect_performance_measures_over_admin, collect_performance_measures_over_station
import GloFAS.GloFAS_prep.configuration as cfg
from comparison.observation.HydroImpact import transform_hydro
from comparison.observation.thresholds import Q_Gumbel_fit_RP
class Visualizer: 
    def __init__(self, DataDir, vector_adminMap, leadtimes, RPsyr, percentiles):
        self.models = ['GloFAS', 'GoogleFloodHub', 'PTM'] # is PTM best way to refer to the current trigger model in the EAP? 
        self.colors = ['cornflowerblue', 'salmon','darkgreen']
        self.markersize = 12
        self.admin_colors = [
            "gold", "cornflowerblue", "#ff7f00", "#4daf4a", "#e41a1c", "#984ea3", 
            "#a65628", "#f781bf", "#999999", "#a6cee3", 
            "#009E73", "#0072B2"
            ]#['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', ] # adjust pls if you want 
        self.linestyles = ['-', '--','-.' ]
        self.markerstyles =['o','v', 's']
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
        metrics = ['pod', 'far']

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

    def performance_over_param(self, admin_unit, data, standard_RP=5, standard_leadtime=168): 
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(f'Performance Metrics for {admin_unit}', fontsize=16)
        leadtimes = data['leadtimes']
        lt_idx = leadtimes.index(standard_leadtime)
        return_periods = data['return_periods']
        RP_idx = return_periods.index(standard_RP)
        leadtimes_x_lim = [min(leadtimes), max(leadtimes)]
        RP_x_lim = [min(return_periods)-0.5, max(return_periods)+0.5]
        
        # Plot 1: POD against leadtime
        ax = axs[0, 0]
        for model, color in zip(self.models, self.colors):
            # 2 in return period is 5yrs of return period !! 0=1.5 1=2, 2= 5
            ax.plot(leadtimes, data['POD'][model]['Observation'][RP_idx,:], color=color,marker=self.markerstyles[0], linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.plot(leadtimes, data['POD'][model]['Impact'][RP_idx,:], color=color, marker=self.markerstyles[1], linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Leadtime (hours)')
        ax.set_ylabel('POD')
        #ax.set_title('POD vs Leadtime')
        ax.set_xlim (leadtimes_x_lim )
        ax.set_ylim([-0.05,1.05])
        ax.text (30, 0.97, f'Return period={standard_RP:.1f} years')
        #ax.legend()
        #ax.grid(True)

        # Plot 2: POD against return period
        ax = axs[0, 1]
        for model, color in zip(self.models, self.colors):
            # 4th index in leadtime index is 168 hours, 7 days 
            ax.plot(return_periods, data['POD'][model]['Observation'][:,lt_idx], color=color,marker=self.markerstyles[0], linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.plot(return_periods, data['POD'][model]['Impact'][:,lt_idx], color=color, marker=self.markerstyles[1], linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('POD')
        #ax.set_title('POD vs Return Period')
        ax.set_xlim (RP_x_lim)
        ax.set_ylim([-0.05,1.05])
        ax.text (1.6, 0.97, f'Leadtime={standard_leadtime:.0f} hours ({standard_leadtime/24:.0f} days)')
        ax.legend(bbox_to_anchor=(1.9,1.0), loc='upper right')
        #ax.grid(True)

        # Plot 3: FAR against leadtime
        ax = axs[1, 0]
        for model, color in zip(self.models, self.colors):
            ax.plot ()
            ax.plot(leadtimes, data['FAR'][model]['Observation'][RP_idx,:], color=color,marker=self.markerstyles[0], linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.plot(leadtimes, data['FAR'][model]['Impact'][RP_idx,:], color=color,marker=self.markerstyles[1], linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Leadtime (hours)')
        ax.set_ylabel('FAR')
        #ax.set_title('FAR vs Leadtime')
        ax.set_xlim (leadtimes_x_lim )
        ax.set_ylim([-0.05,1.05])
        ax.text (30, 0.97, f'Return period={standard_RP:.1f} years')
        #ax.legend()
        #ax.grid(True)

        # Plot 4: FAR against return period
        ax = axs[1, 1]
        for model, color in zip(self.models, self.colors):
            ax.plot(return_periods, data['FAR'][model]['Observation'][:,lt_idx], color=color, marker=self.markerstyles[0], linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.plot(return_periods, data['FAR'][model]['Impact'][:,lt_idx], color=color, marker=self.markerstyles[1], linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('FAR')
        #ax.set_title('FAR vs Return Period')
        ax.set_xlim (RP_x_lim)
        ax.set_ylim([-0.05,1.05])
        ax.text (1.05, 0.97, f'Leadtime={standard_leadtime:.0f} hours ({standard_leadtime/24:.0f} days)')
        #ax.legend()
        #ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filePath = f'{self.DataDir}/comparison/results/performance_metrics_{admin_unit}_RP{standard_RP:.1f}_leadtime{standard_leadtime}.png'
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

        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        fig.suptitle(f'Performance metrics across administrative units over',fontsize=12) # {"lead time" if x_axis == "leadtime" el "return period"}',

        # Determine axis values based on mode
        if x_axis == 'leadtime':
            x_values = self.leadtimes
            x_label = 'Leadtime (hours)'
            x_lim = [min(x_values), max(x_values)]
            if threshold_type=='return_periods':
                standard_idx = self.RPsyr.index(standard_value)
                secondary_text = f'Return period={standard_value:.1f} years'
            elif threshold_type=='percentiles': 
                standard_idx = self.percentiles.index(standard_value)
                secondary_text = f'Percentile={standard_value:.0f}%'
            else: 
                raise ValueError (f"choose threshold_type of type 'return_periods' or 'percentiles', not {threshold_type}")
        elif x_axis =='percentile': 
            x_values = self.percentiles 
            x_label = 'Percentile (%)'
            standard_idx = self.leadtimes.index(standard_value)
            x_lim = [min(x_values) - 0.5, max(x_values) + 0.5]
            secondary_text = f'Leadtime={standard_value:.0f} hours ({standard_value / 24:.0f} days)'
        elif x_axis == 'return_period':
            x_values = self.RPsyr
            x_label = 'Return Period (years)'
            x_lim = [min(x_values) - 0.5, max(x_values) + 0.5]
            standard_idx = self.leadtimes.index(standard_value)
            secondary_text = f'Leadtime={standard_value:.0f} hours ({standard_value / 24:.0f} days)'
        else: 
            raise ValueError (f"choose x_axis 'return_period', 'percentile' or 'leadtime', not {x_axis}")

        # Plot POD
        ax = axs[0]
        for spatial_unit, color in zip(spatial_units, self.admin_colors):
            for model, marker, linestyle in zip(self.models, self.markerstyles, self.linestyles):
                if x_axis == 'leadtime':
                    y_values = data_all_admin[spatial_units.index(spatial_unit)][1]['POD'][model]['Observation'][threshold_type][standard_idx, :]
                else:
                    y_values = data_all_admin[spatial_units.index(spatial_unit)][1]['POD'][model]['Observation'][threshold_type][:, standard_idx]
                ax.plot(x_values, y_values, color=color, linestyle=linestyle, marker=marker, markersize=self.markersize)
        ax.set_xlabel(x_label)
        ax.set_ylabel('POD')
        ax.set_xlim(x_lim)
        ax.set_ylim([-0.05, 1.05])
        ax.fill_between(x_lim, ax.get_ylim()[0], self.POD_threshold, color='red', alpha=0.3)
        ax.text(x_lim[0] + ((x_lim[1]-x_lim[0])/100), 1.10, secondary_text)

        # Plot FAR
        ax = axs[1]
        for spatial_unit, color in zip(spatial_units, self.admin_colors):
            for model, marker, linestyle in zip(self.models, self.markerstyles, self.linestyles):
                if x_axis == 'leadtime':
                    y_values = data_all_admin[spatial_units.index(spatial_unit)][1]['FAR'][model]['Observation'][threshold_type][standard_idx, :]
                else:
                    y_values = data_all_admin[spatial_units.index(spatial_unit)][1]['FAR'][model]['Observation'][threshold_type][:, standard_idx]
                ax.plot(x_values, y_values, color=color, linestyle=linestyle, marker=marker, markersize=self.markersize)
        ax.set_xlabel(x_label)
        ax.set_ylabel('FAR')
        ax.set_xlim(x_lim)
        ax.set_ylim([-0.05, 1.05])
        ax.fill_between(x_lim, ax.get_ylim()[1], self.FAR_threshold, color='red', alpha=0.3)
        ax.text(x_lim[0] + ((x_lim[1]-x_lim[0])/100), 1.10, secondary_text)

        # Custom Legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='black', marker=self.markerstyles[0], linestyle=self.linestyles[0], label='GloFAS'),
            #Line2D([0], [0], color='black', marker=self.markerstyles[1], linestyle=self.linestyles[1], label='Google Flood Hub'),
            Line2D([0], [0], color='black', marker=self.markerstyles[2], linestyle=self.linestyles[2], label='PTM'),
        ]
        color_lines = [Line2D([0], [0], color=color, lw=2, label=unit) for color, unit in zip(self.admin_colors, spatial_units)]
        fig.legend(handles=custom_lines + color_lines,
                loc='lower center', ncol=4, fontsize='small', frameon=False)

        plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Adjusted spacing for legend
        file_path = f'{self.DataDir}/comparison/results/performance_metrics_over{x_axis}_all_admin_otherv{standard_value}.png'
        plt.savefig(file_path)
        plt.show()


    def plot_flood_and_impact_events(self, df_glofas, df_impact, df_obs, stationname, admin_unit, leadtime, return_period, start_time, end_time, threshold_glofas, threshold_obs ):
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
        print (df_impact_bamako_events.head)
        # Convert start and end times to datetime for filtering
        start_time = pd.to_datetime(start_time, format="%Y-%m-%d")
        end_time = pd.to_datetime(end_time, format="%Y-%m-%d")
        # Filter observed and predicted discharge for the time range
        df_obs = df_obs.loc[(df_obs.index >= start_time) & (df_obs.index <= end_time)]
        df_glofas = df_glofas.loc[(df_glofas.index >= start_time) & (df_glofas.index <= end_time)]
        
        # Plot the data
        plt.figure(figsize=(12, 6))

        # Plot observed discharge
        plt.plot(df_obs.index, df_obs['Value'], label='Observed Discharge', color='blue', linewidth=1)

        # Plot predicted discharge
        plt.plot(df_glofas.index, df_glofas['percentile_40.0'], label=f'Predicted discharge by GloFAS with leadtime of {leadtime} hours', color='green', linestyle='-', linewidth=1)
        plt.axhline(threshold_obs, color='blue', linestyle=':', linewidth=2, label=f'{return_period}-year Return Period of observational data')
        plt.axhline(threshold_glofas, color='green', linestyle=':', linewidth=2, label=f'{return_period}-year Return Period of GloFAS data ')


        # Add impact events as translucent red blocks with a single legend entry
        legend_shown = False
        for _, row in df_impact_bamako_events.iterrows():
            start = pd.to_datetime(row['Start Date'])
            end = pd.to_datetime(row['End Date'])
            if start >= start_time and end <= end_time:  # Only include events within the specified time range
                label = 'Impact Event' if not legend_shown else ""
                plt.axvspan(start, end, color='red', alpha=0.3, label=label)
                legend_shown = True
        # Add horizontal line for threshold (e.g., 5-year return period)

        # Customize plot
        plt.title(f'Discharge at station {stationname} and impact events in {admin_unit}')
        plt.xlabel('Date')
        plt.ylabel('Discharge (m³/s)')
        plt.legend(loc='lower left')
        #plt.grid(True)
        plt.tight_layout()
        
        # Show plot
        plt.show()
if __name__ =='__main__': 
    vis = Visualizer(cfg.DataDir, cfg.admPath, cfg.leadtimes, cfg.RPsyr, cfg.percentiles)
    # BasinName = 'Niger'
    # StationName = 'Bamako'
    # CorrespondingAdminUnit = 'Bamako'
    
    # leadtime = 168
    # return_period = 2 # year 
    # df_obs = transform_hydro(f"{cfg.DataDir}/DNHMali_2019/Q_stations/{BasinName}_{StationName}.csv",cfg.startYear, cfg.endYear)
    # df_glofas = pd.read_csv (f"{cfg.stationsDir}/GloFAS_Q/timeseries/discharge_timeseries_{StationName}_{leadtime}.csv",
    #                             )
    # df_glofas['ValidTime'] = pd.to_datetime(df_glofas["ValidTime"], format="%Y-%m-%d")
    # # Check for any invalid dates
    # if df_glofas['ValidTime'].isnull().any():
    #     print("Warning: Some dates in 'date_col' could not be parsed and are set to NaT.")
    # # Set the index to the datetime column
    # df_glofas.set_index('ValidTime', inplace=True)
    # df_glofas.sort_index(inplace=True)
    # df_impact = pd.read_csv (cfg.impact_csvPath, delimiter=';', header=0)
    # RP_glofas = Q_Gumbel_fit_RP(df_glofas, return_period, 'percentile_40.0')
    # RP_obs = Q_Gumbel_fit_RP (df_obs, return_period, 'Value')
    # 
    # vis.plot_flood_and_impact_events(df_glofas, 
    #                                     df_impact, 
    #                                     df_obs, 
    #                                     StationName,
    #                                     CorrespondingAdminUnit, 
    #                                     leadtime,
    #                                     return_period,
    #                                     '2004-07-01', 
    #                                     '2019-01-01', RP_glofas, RP_obs)
    comparisonTypes = ['Observation']
    models = ['GloFAS', 'PTM']
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
    station_names = [ 'GUELELINKORO','BAMAKO', 'BANANKORO', 'KOULIKORO','MOPTI', 'DIRE', 'ANSONGO', 'GAO', 'SOFARA', 'DOUNA', 'BOUGOUNI', 'PANKOUROU','BAFING MAKANA', 'KAYES']
    #admin_units = [''][#['BLA', 'SAN','KIDAL', 'TOMINIAN', 'KANGABA', 'KOULIKORO', 'KOLONDIEBA', 'MOPTI', 'BAMAKO', 'SIKASSO', 'SEGOU', 'KATI']
    for leadtime in cfg.leadtimes:
        vis.performance_metrics(station_names, spatial_unit_type='StationNames', x_axis='percentile', standard_value=leadtime, threshold_type='percentiles')
        vis.performance_metrics(station_names, spatial_unit_type='StationNames', x_axis='return_period', standard_value=leadtime, threshold_type='return_periods')
    for RPyr in cfg.RPsyr:
        vis.performance_metrics (station_names, spatial_unit_type='StationNames', x_axis='leadtime', standard_value=RPyr)
    for percentile in cfg.percentiles: 
        vis.performance_metrics (station_names, spatial_unit_type='StationNames', x_axis='leadtime', standard_value=percentile, threshold_type='percentiles')
    # for admin_unit in admin_units:
    #     data = collect_performance_measures(admin_unit, cfg.DataDir, cfg.leadtimes, cfg.RPsyr)
    #     for leadtime in cfg.leadtimes: 
    #         for RPyr in cfg.RPsyr:
    #             vis.performance_over_param(admin_unit, data, standard_RP=RPyr, standard_leadtime=leadtime)
            

# to do : 
## add lines between points 
## general layouts 