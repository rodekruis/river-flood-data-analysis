# src/extract/__init__.py

__version__ = '0.0.0' # MAJOR.MINOR.PATCH versioning
__author__ = 'valentijn7' # GitHub username

print('\nRunning __init__.py for GoogleFloodHub-data-extractor')

from .parse import validate_args
from .exceptions import handle_exception
from .getters import get_API_key
from .getters import get_json_file
from .export import extract_country_data_for_time_delta
from .call_ListGauges import get_ListGauges
from .call_GetGaugeModel import get_GetGaugeModel
from .call_QueryGaugeForecasts import get_QueryGaugeForecasts

print('GoogleFloodHub-data-extractor initialized\n')