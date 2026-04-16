import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence
import GloFAS.GloFAS_prep.config_comp as cfg
from comparison.visualization.plotting_utils import annotate_plot_point, add_hline_across_subplots
def plot_POD_vs_action_lifetime(
    dfs: List[pd.DataFrame],
    names: List[str],
    save_figure: bool = False,
    lead_time: int = 7,
):
    fig, ax = plt.subplots(figsize=(4 * 1.618, 4))
    fs = 12

    colours = ['blue', 'red', 'magenta', 'black']
    markers = ['^', 's', 'D', 'o', 'v', 'x']

    for idx, (df, label) in enumerate(zip(dfs, names)):
        mk  = markers[idx % len(markers)]
        col = colours[idx % len(colours)]

        x = df['actionlifetime'].values
        y = df['POD'].values

        mask = (~np.isnan(y)) & (y != 0)

        ax.plot(x[mask], y[mask], marker=mk, color=col, label=label)

    # Labels
    ax.set_xlabel('Action lifetime (days)', fontsize=fs)
    ax.set_ylabel('mean POD [-]', fontsize=fs)

    # Limits and ticks
    ax.set_ylim(0, 1)
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    # Styling
    ax.legend(loc='upper left', fontsize=fs, frameon=False)
    ax.text(
        0.00, 1.05,
        f'Lead time = {lead_time} days',
        ha='left', va='top',
        transform=ax.transAxes,
        fontsize=fs
    )

    # Horizontal performance lines
    ax.axhline(0.5, color='#EB5B00', linestyle='--', linewidth=1.0)
    ax.text(ax.get_xlim()[1], 0.5, "'Acceptable'", va='bottom', ha='right', fontsize=fs)

    ax.axhline(0.6, color='#0D4715', linestyle='--', linewidth=1.0)
    ax.text(ax.get_xlim()[1], 0.6, "'Good'", va='bottom', ha='right', fontsize=fs)

    plt.tight_layout()

    if save_figure:
        plt.savefig(
            f'../data/figures/results_POD_vs_action_lifetime.pdf',
            bbox_inches='tight',
            dpi=300
        )

    plt.show()


if __name__ == '__main__':
    percentile = 95
    leadtime = 168
    metric = 'POD'
    comparisonType = 'Observation'
    reference='OBS'
    model = 'GloFAS'
    dfs = []
    row_to_plot = 'mean'
    glofas_df = pd.DataFrame(columns = ['POD', 'FAR'], index = [''])

    rows = []
    action_lifetimes = []
    pod_values = []
    for actionlifetime in [2,3,4,5,6,7,8,9,10]:
        al_df = pd.read_csv(
            f"{cfg.DataDir}/{model}/{comparisonType}/{model}_vs_{reference}_{leadtime}lt_{percentile:.1f}_{actionlifetime}al.csv"
        )

        mask_pod = al_df['pod'].isna() & (al_df['far'].notna() | al_df['FN']>0)
        al_df.loc[mask_pod, 'pod'] = 0

        mask_far = al_df['far'].isna() & (al_df['pod'].notna() | al_df['TP'].notna())
        al_df.loc[mask_far, 'far'] = 0

        POD_glofas = al_df['pod'].mean()
        FAR_glofas = al_df['far'].mean()
        rows.append({
            'actionlifetime': actionlifetime,
            'POD': POD_glofas,
            'FAR': FAR_glofas,
        })

    glofas_df = pd.DataFrame(rows)

    plot_POD_vs_action_lifetime(
        dfs=[glofas_df],
        names=[model],
        lead_time=int(leadtime / 24)
    )
                            