import cmcrameri.cm as cmc #change colormaps
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
import matplotlib.pyplot as plt

class Visualizer: 
    def __init__(self, DataDir, vector_adminMap):
        self.models = ['GloFAS', 'GoogleFloodHub', 'EAP'] # is EAP best way to refer to the current trigger model in the EAP? 
        self.colors = ['cornflowerblue', 'salmon','darkgreen'] # adjust pls if you want 
        self.linestyles = ['-', '--']
        self.DataDir=DataDir
        self.gdf_shape=checkVectorFormat(vector_adminMap, shapeType='polygon')

    def map_performance(self, scores_by_commune_gdf, RPyr, leadtime):
        """
        Visualize performance metrics (POD, FAR, CSI, POFD, Accuracy, Precision) on separate maps.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        plt.suptitle(f'Performance Metrics for Flood Prediction (RP{RPyr:.1f} Year, Lead Time: {leadtime/24:.0f} Days)')

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
        filePath = f'{self.DataDir}/performance_metrics_RP{RPyr:.1f}_yr_leadtime{leadtime/24:.0f}.png'
        plt.savefig(filePath)
        plt.show()

    def performance_over_param(self, admin_unit, data): 
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Metrics for {admin_unit}', fontsize=16)
        
        # Plot 1: POD against leadtime
        ax = axs[0, 0]
        for model, color in zip(self.models, self.colors):
            ax.plot(leadtime, data['POD'][model]['observation'], color=color, linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.plot(leadtime, data['POD'][model]['impact'], color=color, linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Leadtime (hours)')
        ax.set_ylabel('POD')
        ax.set_title('POD vs Leadtime')
        ax.legend()
        ax.grid(True)

        # Plot 2: POD against return period
        ax = axs[0, 1]
        for model, color in zip(self.models, self.colors):
            ax.plot(return_period, data['POD'][model]['observation'], color=color, linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.plot(return_period, data['POD'][model]['impact'], color=color, linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('POD')
        ax.set_title('POD vs Return Period')
        ax.legend()
        ax.grid(True)

        # Plot 3: FAR against leadtime
        ax = axs[1, 0]
        for model, color in zip(self.models, self.colors):
            ax.plot(leadtime, data['FAR'][model]['observation'], color=color, linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.plot(leadtime, data['FAR'][model]['impact'], color=color, linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Leadtime (hours)')
        ax.set_ylabel('FAR')
        ax.set_title('FAR vs Leadtime')
        ax.legend()
        ax.grid(True)

        # Plot 4: FAR against return period
        ax = axs[1, 1]
        for model, color in zip(self.models, self.colors):
            ax.plot(return_period, data['FAR'][model]['observation'], color=color, linestyle=self.linestyles[0], label=f'{model} (Obs)')
            ax.plot(return_period, data['FAR'][model]['impact'], color=color, linestyle=self.linestyles[1], label=f'{model} (Impact)')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel('FAR')
        ax.set_title('FAR vs Return Period')
        ax.legend()
        ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()