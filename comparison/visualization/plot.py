import cmcrameri.cm as cmc #change colormaps
import pandas as pd
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
from GloFAS.GloFAS_prep.text_formatter import capitalize
import matplotlib.pyplot as plt
from comparison.collect_for_administrative_unit import collect_performance_measures
import GloFAS.GloFAS_prep.configuration as cfg
from comparison.observation.HydroImpact import transform_hydro
from comparison.observation.thresholds import Q_Gumbel_fit_RP
class Visualizer: 
    def __init__(self, DataDir, vector_adminMap):
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
        self.comparisonTypes = ['Observation', 'Impact']
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

    def performance_over_leadtime_all(self, admin_units, standard_RP=5): #, standard_leadtime=168): 
        data_all_admin = []
        for admin_unit in admin_units:
            data = collect_performance_measures(admin_unit, self.DataDir, cfg.leadtimes, cfg.RPsyr)
            data_all_admin.append((admin_unit, data))

        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        fig.suptitle('Performance metrics across administrative units over return period', fontsize=12)
        return_periods = cfg.RPsyr
        RP_idx = return_periods.index(standard_RP)
        leadtimes = cfg.leadtimes
        leadtimes_x_lim = [min(leadtimes), max(leadtimes)]

        # Plot 1: POD against leadtime
        ax = axs[0]
        for admin_unit, color in zip(admin_units, self.admin_colors):
            for model, marker in zip(self.models, self.markerstyles):
                for comparison_type, linestyle in zip(self.comparisonTypes, self.linestyles):
                    ax.plot(leadtimes, 
                            data_all_admin[admin_units.index(admin_unit)][1]['POD'][model][comparison_type][RP_idx, :], 
                            color=color, 
                            linestyle=linestyle, 
                            marker=marker,
                            markersize=self.markersize,
                            label=f'{admin_unit}, {model}, {comparison_type}')
        ax.set_xlabel('Leadtime (hours)')
        ax.set_ylabel('POD')
        ax.fill_between(leadtimes, ax.get_ylim()[0], self.POD_threshold, color='red', alpha=0.3)
        ax.set_xlim(leadtimes_x_lim)
        ax.set_ylim([-0.05, 1.05])
        ax.text(24, 1.10, f'Return period={standard_RP:.1f} years')


        # Plot 3: FAR against leadtime
        ax = axs[1]
        for admin_unit, color in zip(admin_units, self.admin_colors):
            for model, marker in zip(self.models, self.markerstyles):
                for comparison_type, linestyle in zip(self.comparisonTypes, self.linestyles):
                    ax.plot(leadtimes, 
                            data_all_admin[admin_units.index(admin_unit)][1]['FAR'][model][comparison_type][RP_idx, :], 
                            color=color, 
                            linestyle=linestyle, 
                            marker=marker,
                            markersize=self.markersize)
        ax.set_xlabel('Leadtime (hours)')
        ax.set_ylabel('FAR')
        ax.fill_between(leadtimes, ax.get_ylim()[1], self.FAR_threshold, color='red', alpha=0.3)
        ax.set_xlim(leadtimes_x_lim)
        ax.set_ylim([-0.05, 1.05])
        ax.text(24, 1.10, f'Return period={standard_RP:.1f} years')

        # Custom Legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='black', linestyle=self.linestyles[0], label='Observation'),
            Line2D([0], [0], color='black', linestyle=self.linestyles[1], label='Impact'),
            #Line2D([0], [0], color='black', linestyle=self.linestyles[2], label='PTM'),
            Line2D([0], [0], color='black', marker=self.markerstyles[0], linestyle='None', label='GloFAS'),
            Line2D([0], [0], color='black', marker=self.markerstyles[1], linestyle='None', label='Google Flood Hub'),
            Line2D([0], [0], color='black', marker=self.markerstyles[2], linestyle='None', label='PTM')
        ]
        color_lines = [Line2D([0], [0], color=color, lw=2, label=unit) for color, unit in zip(self.admin_colors, admin_units)]
        fig.legend(handles=custom_lines + color_lines, 
                loc='lower center', 
                ncol=4, 
                fontsize='small', 
                frameon=False)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        filePath = f'{self.DataDir}/comparison/results/performance_metrics_all_admin_RP{standard_RP:.1f}.png'
        plt.savefig(filePath)
        plt.show()

    def performance_over_return_period_all(self, admin_units, standard_leadtime=168): 
        data_all_admin = []
        for admin_unit in admin_units:
            data = collect_performance_measures(admin_unit, self.DataDir, cfg.leadtimes, cfg.RPsyr)
            data_all_admin.append((admin_unit, data))

        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        fig.suptitle('Performance metrics across administrative units over lead time', fontsize=12)
        leadtimes = cfg.leadtimes
        lt_idx = leadtimes.index(standard_leadtime)
        return_periods = cfg.RPsyr
        RP_x_lim = [min(return_periods) - 0.5, max(return_periods) + 0.5]


        # Plot 2: POD against return period
        ax = axs[0]
        for admin_unit, color in zip(admin_units, self.admin_colors):
            for model, marker in zip(self.models, self.markerstyles):
                for comparison_type, linestyle in zip(self.comparisonTypes, self.linestyles):
                    ax.plot(return_periods, 
                            data_all_admin[admin_units.index(admin_unit)][1]['POD'][model][comparison_type][:, lt_idx], 
                            color=color, 
                            linestyle=linestyle, 
                            marker=marker,
                            markersize=self.markersize)
        
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('POD')
        ax.set_xlim(RP_x_lim)
        ax.set_ylim([-0.05, 1.05])
        ax.fill_between(RP_x_lim, ax.get_ylim()[0], self.POD_threshold, color='red', alpha=0.3)
        ax.text(1.5, 1.10, f'Leadtime={standard_leadtime:.0f} hours ({standard_leadtime / 24:.0f} days)')


        # Plot 4: FAR against return period
        ax = axs[1]
        for admin_unit, color in zip(admin_units, self.admin_colors):
            for model, marker in zip(self.models, self.markerstyles):
                for comparison_type, linestyle in zip(self.comparisonTypes, self.linestyles):
                    ax.plot(return_periods, 
                            data_all_admin[admin_units.index(admin_unit)][1]['FAR'][model][comparison_type][:, lt_idx], 
                            color=color, 
                            linestyle=linestyle, 
                            marker=marker,
                            markersize=self.markersize)
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('FAR')
        ax.set_xlim(RP_x_lim)
        ax.set_ylim([-0.05, 1.05])
        ax.fill_between(RP_x_lim, ax.get_ylim()[1], self.FAR_threshold, color='red', alpha=0.3)
        ax.text(1.5, 1.10, f'Leadtime={standard_leadtime:.0f} hours ({standard_leadtime / 24:.0f} days)')

        # Custom Legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='black', linestyle=self.linestyles[0], label='Observation'),
            Line2D([0], [0], color='black', linestyle=self.linestyles[1], label='Impact'),
            #Line2D([0], [0], color='black', linestyle=self.linestyles[2], label='PTM'),
            Line2D([0], [0], color='black', marker=self.markerstyles[0], linestyle='None', label='GloFAS'),
            Line2D([0], [0], color='black', marker=self.markerstyles[1], linestyle='None', label='Google Flood Hub'),
            Line2D([0], [0], color='black', marker=self.markerstyles[2], linestyle='None', label='PTM')
        ]
        color_lines = [Line2D([0], [0], color=color, lw=2, label=unit) for color, unit in zip(self.admin_colors, admin_units)]
        fig.legend(handles=custom_lines + color_lines, 
                loc='lower center', 
                ncol=4, 
                fontsize='small', 
                frameon=False)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        filePath = f'{self.DataDir}/comparison/results/performance_metrics_all_admin_leadtime{standard_leadtime}.png'
        plt.savefig(filePath)
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
    BasinName = 'Niger'
    StationName = 'Bamako'
    CorrespondingAdminUnit = 'Bamako'
    
    leadtime = 168
    return_period = 2 # year 
    df_obs = transform_hydro(f"{cfg.DataDir}/DNHMali_2019/Q_stations/{BasinName}_{StationName}.csv",cfg.startYear, cfg.endYear)
    df_glofas = pd.read_csv (f"{cfg.stationsDir}/GloFAS_Q/timeseries/discharge_timeseries_{StationName}_{leadtime}.csv",
                                )
    df_glofas['ValidTime'] = pd.to_datetime(df_glofas["ValidTime"], format="%Y-%m-%d")
    # Check for any invalid dates
    if df_glofas['ValidTime'].isnull().any():
        print("Warning: Some dates in 'date_col' could not be parsed and are set to NaT.")
    # Set the index to the datetime column
    df_glofas.set_index('ValidTime', inplace=True)
    df_glofas.sort_index(inplace=True)
    df_impact = pd.read_csv (cfg.impact_csvPath, delimiter=';', header=0)
    RP_glofas = Q_Gumbel_fit_RP(df_glofas, return_period, 'percentile_40.0')
    RP_obs = Q_Gumbel_fit_RP (df_obs, return_period, 'Value')
    vis = Visualizer(cfg.DataDir, cfg.admPath)
    vis.plot_flood_and_impact_events(df_glofas, 
                                        df_impact, 
                                        df_obs, 
                                        StationName,
                                        CorrespondingAdminUnit, 
                                        leadtime,
                                        return_period,
                                        '2004-07-01', 
                                        '2019-01-01', RP_glofas, RP_obs)
    # comparisonTypes = ['Observation', 'Impact']
    # models = ['GloFAS', 'PTM']
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
    # admin_units = [ 'KOULIKORO']
    # admin_units = ['BLA', 'SAN','KIDAL', 'TOMINIAN', 'KANGABA', 'KOULIKORO', 'KOLONDIEBA', 'MOPTI', 'BAMAKO', 'SIKASSO', 'SEGOU', 'KATI']
    # for leadtime in cfg.leadtimes:
    #     vis.performance_over_return_period_all (admin_units, leadtime)
    # for RPyr in cfg.RPsyr:
    #     vis.performance_over_leadtime_all (admin_units, standard_RP=RPyr)
    # for admin_unit in admin_units:
    #     data = collect_performance_measures(admin_unit, cfg.DataDir, cfg.leadtimes, cfg.RPsyr)
    #     for leadtime in cfg.leadtimes: 
    #         for RPyr in cfg.RPsyr:
    #             vis.performance_over_param(admin_unit, data, standard_RP=RPyr, standard_leadtime=leadtime)
            

# to do : 
## add lines between points 
## general layouts 