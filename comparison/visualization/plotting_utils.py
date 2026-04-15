from typing import Optional, Sequence

def add_hline_across_subplots(fig, ax_ref, y, ax_left, ax_right, label, **line_kwargs):
    # LLM-generated code:
    import matplotlib.lines as mlines
    display_coords = ax_ref.transData.transform((0, y))
    y_fig = fig.transFigure.inverted().transform(display_coords)[1]
    bbox_left  = ax_left.get_position()
    bbox_right = ax_right.get_position()
    line = mlines.Line2D([bbox_left.x0, bbox_right.x1],
                         [y_fig, y_fig],
                         transform = fig.transFigure, **line_kwargs)
    fig.add_artist(line)

    if label:
        dummy_text = fig.text(0, 0, label, fontsize = 12)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = dummy_text.get_window_extent(renderer=renderer)
        text_width_pixels = bbox.width
        fig_width_pixels = fig.get_figwidth() * fig.dpi
        text_width_frac = text_width_pixels / fig_width_pixels
        dummy_text.remove()
        x_text = bbox_right.x1 - text_width_frac
        fig.text(x_text - 0.005, y_fig + 0.015, label,
                 fontsize = 12, color = line_kwargs['color'])


# LLM-generated code:
def annotate_plot_point(ax, xs, ys, labels, color, place_below=None):
    if place_below is None:
        place_below = set()
    for x, y, lab in zip(xs, ys, labels):
        # normalize for comparison (handle numbers or strings)
        lab_norm = None
        try:
            lab_norm = int(lab)
        except Exception:
            lab_norm = str(lab).strip()

        if (lab_norm in place_below) or (str(lab_norm) in {str(v) for v in place_below}):
            offset = (0, -6); va = 'top'     # put label below the point
        else:
            offset = (0,  6); va = 'bottom'  # default: above the point

        ax.annotate(
            str(lab), (x, y),
            xytext=offset, textcoords='offset points',
            ha='center', va=va,
            fontsize=10, color=color, clip_on=True
        )
        
def get_metrics_for_configuration(
        d: Dict[str, pd.DataFrame], # the dictionary with the results
        type: str,                  # the type of results, e.g. 'GFH_vs_IMPACT'
        row: str,                   # where to get the metric for, i.e. in 'identifier' a location or 'total'
        metric: str,                # the metric to calculate (TP, FP, FN, POD, FAR, precision, f1)
    ) -> pd.DataFrame:
    """ Returns metric dataframe with dimensions (lead_time, return_period) """
    if not any(key.startswith(type) for key in d.keys()):
        raise ValueError("Type must be in the keys of the dictionary")
    if row not in next(iter(d.values()))['identifier'].unique():
        raise ValueError("Row must be in the identifier column of the dataframe")
    if metric not in ['TP', 'FP', 'FN', 'POD', 'FAR', 'precision', 'f1']:
        raise ValueError("Metric must be in ['TP', 'FP', 'FN', 'POD', 'FAR', 'precision', 'f1']")
    if metric == 'f1':
        # ensure the column 'f1' of all dataframes of d uses lowercase 'f1' instead of uppercase 'F1'
        for key in d.keys():
            d[key] = d[key].rename(columns = {'F1': 'f1'})

    lts = ['96lt', '120lt', '144lt', '168lt']
    rps = ['95pc', '98pc', '99pc', '1.5rp', '2rp', '5rp', '10rp']
    df = pd.DataFrame(index = lts, columns = rps)
    
    for lt in lts:
        for rp in rps:
            key = f'{type}_{lt}_{rp}'
            df.loc[lt, rp] = d[key][d[key]['identifier'] == row][metric].values[0]
    return df