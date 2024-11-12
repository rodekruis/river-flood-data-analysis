# src/analyse/__init__.py

__version__ = '0.0.0' # MAJOR.MINOR.PATCH versioning
__author__ = 'valentijn7' # GitHub username

print('\nRunning __init__.py for GoogleFloodHub-data-analyser')

from .getters import get_country_data
from .getters import get_country_polygon
from .getters import get_shape_file
from .transform import convert_df_to_gdf
from .transform import subset_country_gauge_coords
from .transform import make_subset_for_gauge_and_issue_time
from .statistics import get_stats_for_forecast_range
from .plots import map_gauge_coordinates_of_country
from .plots import plot_x_days_of_gauge_forecast_for_issue_time
from .plots import plot_forecast_min_mean_max
from .plots import plot_Niger_river_downstream_flow_stat
from .tests import assert_same_coord_system

print('GoogleFloodHub-data-analyser initialized\n')