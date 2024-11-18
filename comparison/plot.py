import cmcrameri.cm as cmc #change colormaps
from GloFAS.GloFAS_prep.vectorCheck import checkVectorFormat
import matplotlib.pyplot as plt

class Visualizer: 
    def __init__(self, DataDir, vector_adminMap):
        self.DataDir=DataDir
        self.gdf_shape=checkVectorFormat(vector_adminMap, shapeType='polygon')

    def visualize_performance(self, scores_by_commune_gdf, RPyr, leadtime):
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