import cmcrameri.cm as cmc #change colormaps
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
import matplotlib.pyplot as plt
from comparison.collect_for_administrative_unit import collect_performance_measures
import GloFAS.GloFAS_prep.configuration as cfg

class Visualizer: 
    def __init__(self, DataDir, vector_adminMap):
        self.models = ['GloFAS', 'GoogleFloodHub', 'PTM'] # is PTM best way to refer to the current trigger model in the EAP? 
        self.colors = ['cornflowerblue', 'salmon','darkgreen'] # adjust pls if you want 
        self.linestyles = ['-', '--']
        self.markerstyle =['o','v']
        self.DataDir=DataDir
        self.gdf_shape=checkVectorFormat(vector_adminMap, shapeType='polygon')

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

        # Define color maps
        cmap = 'cmc.batlow'
        cmap_r = 'cmc.batlow_r'

        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            # Plot each metric
            self.gdf_shape.plot(ax=ax, color='lightgrey', alpha=0.5)
            if title in ['POD (Probability of Detection)', 'CSI (Critical Success Index)', 'Accuracy', 'Precision']:
                scores_by_commune_gdf.plot(column=metric, cmap=cmap_r, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, vmin=0, vmax=1)
            else:
                scores_by_commune_gdf.plot(column=metric, cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, vmin=0, vmax=1)
            
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
        cmap_pod = 'cmc.batlow_r'  # Reversed for POD
        cmap_far = 'cmc.batlow'    # Default for FAR

        for ax, metric, title, cmap in zip(axes, metrics, titles, [cmap_pod, cmap_far]):
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
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
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
            ax.scatter(leadtimes, data['POD'][model]['Observation'][RP_idx,:], color=color, marker=self.markerstyle[0], linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.scatter(leadtimes, data['POD'][model]['Impact'][RP_idx,:], color=color, marker=self.markerstyle[1], linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Leadtime (hours)')
        ax.set_ylabel('POD')
        ax.set_title('POD vs Leadtime')
        ax.set_xlim (leadtimes_x_lim )
        ax.set_ylim([-0.05,1.05])
        ax.text (72.5, 0.97, f'Return period={standard_RP:.1f} years')
        ax.legend()
        #ax.grid(True)

        # Plot 2: POD against return period
        ax = axs[0, 1]
        for model, color in zip(self.models, self.colors):
            # 4th index in leadtime index is 168 hours, 7 days 
            ax.scatter(return_periods, data['POD'][model]['Observation'][:,lt_idx], color=color,marker=self.markerstyle[0], linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.scatter(return_periods, data['POD'][model]['Impact'][:,lt_idx], color=color, marker=self.markerstyle[1], linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('POD')
        ax.set_title('POD vs Return Period')
        ax.set_xlim (RP_x_lim)
        ax.set_ylim([-0.05,1.05])
        ax.text (1.6, 0.97, f'Leadtime={standard_leadtime:.0f} hours ({standard_leadtime/24:.0f} days)')
        ax.legend()
        #ax.grid(True)

        # Plot 3: FAR against leadtime
        ax = axs[1, 0]
        for model, color in zip(self.models, self.colors):
            ax.scatter(leadtimes, data['FAR'][model]['Observation'][RP_idx,:], color=color, marker=self.markerstyle[0], linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.scatter(leadtimes, data['FAR'][model]['Impact'][RP_idx,:], color=color,marker=self.markerstyle[1], linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Leadtime (hours)')
        ax.set_ylabel('FAR')
        ax.set_title('FAR vs Leadtime')
        ax.set_xlim (leadtimes_x_lim )
        ax.set_ylim([-0.05,1.05])
        ax.text (72.5, 0.97, f'Return period={standard_RP:.1f} years')
        ax.legend()
        #ax.grid(True)

        # Plot 4: FAR against return period
        ax = axs[1, 1]
        for model, color in zip(self.models, self.colors):
            ax.scatter(return_periods, data['FAR'][model]['Observation'][:,lt_idx], color=color, marker=self.markerstyle[0], linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.scatter(return_periods, data['FAR'][model]['Impact'][:,lt_idx], color=color, marker=self.markerstyle[1], linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('FAR')
        ax.set_title('FAR vs Return Period')
        ax.set_xlim (RP_x_lim)
        ax.set_ylim([-0.05,1.05])
        ax.text (1.6, 0.97, f'Leadtime={standard_leadtime:.0f} hours ({standard_leadtime/24:.0f} days)')
        ax.legend()
        #ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filePath = f'{self.DataDir}/comparison/results/performance_metrics_{admin_unit}_RP{standard_RP:.1f}_leadtime{standard_leadtime}.png'
        plt.savefig(filePath)
        plt.show()

if __name__ =='__main__': 
    vis = Visualizer(cfg.DataDir, cfg.admPath)
    comparisonTypes = ['Observation', 'Impact']
    # for comparisonType in comparisonTypes:
    #     for leadtime in cfg.leadtimes: 
    #         for RPyr in cfg.RPsyr:
    #             scores_path = f"{cfg.DataDir}/GloFAS/{comparisonType}/scores_byCommuneRP{RPyr:.1f}_yr_leadtime{leadtime:.0f}.gpkg"
    #             scores_by_commune_gdf = checkVectorFormat(scores_path)
    #             vis.map_pod_far(scores_by_commune_gdf, RPyr, leadtime, comparisonType, 'GloFAS')
    # admin_units = [ 'KOULIKORO', 'SEGOU', 'KATI']
    admin_units = ['BLA', 'SAN','KIDAL', 'TOMINIAN', 'KANGABA', 'KOULIKORO', 'KOLONDIEBA', 'MOPTI', 'BAMAKO', 'SIKASSO', 'SEGOU', 'KATI']
    for admin_unit in admin_units:
        data = collect_performance_measures(admin_unit, cfg.DataDir, cfg.leadtimes, cfg.RPsyr)
        vis.performance_over_param(admin_unit, data, standard_RP=2.0, standard_leadtime=96)

# to do : 
## add lines between points 
## general layouts 