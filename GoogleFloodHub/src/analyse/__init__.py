# src/analyse/__init__.py

__version__ = '0.0.0' # MAJOR.MINOR.PATCH versioning
__author__ = 'valentijn7' # GitHub username

print('\nRunning __init__.py for GoogleFloodHub-data-analyser')

from .datasets import add_quality_verified_flag
from .datasets import export_country_gauge_coords
from .datasets import assign_coords_to_datasets
from .datasets import assign_admin_unit_to_datasets
from .datasets import aggregate_per_admin_unit
from .datasets import aggregate_or_load_per_admin_unit
from .datasets import select_lead_time
from .events import export_dict_flood_events_to_csv
from .events import create_flood_events
from .events import combine_dict_events_to_df
from .getters import get_country_data
from .getters import get_country_polygon
from .getters import get_shape_file
from .getters import get_severity_levels
from .getters import get_datasets_unit_with_most_gauges
from .getters import get_datasets_for_xth_admin_unit
from .statistics import count_gauge_ids
from .statistics import count_admin_units
from .statistics import get_stats_for_forecast_range
from .statistics import z_normalise
from .statistics import calculate_metrics
from .statistics import match_events_and_get_metrics
from .statistics import export_results
from .thresholds import add_thresholds_to_dataset
from .thresholds import add_RPs_and_percentiles
from .transform import convert_df_to_gdf
from .transform import subset_country_gauge_coords
from .transform import make_subset_for_gauge_and_issue_time
from .transform import export_all_results_to_geojson
from .plots import map_gauge_coordinates_of_country
from .plots import plot_x_days_of_gauge_forecast_for_issue_time
from .plots import plot_forecast_min_mean_max
from .plots import plot_Niger_river_downstream_flow_stat
from .plots import plot_reforecast
from .plots import plot_reanalysis
from .plots import add_return_periods
from .plots import plot_aggregated_reforecast
from .tests import assert_same_coord_system
from .tests import assure_all_thresholds_added
from .tests import assure_attributes_assigned

print('GoogleFloodHub-data-analyser initialized\n')