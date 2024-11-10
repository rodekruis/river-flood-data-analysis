# src/extract/exceptions.py

import json
import requests


class GaugesNotAvailableError(Exception):
    """
    Raised when no gauges are available for the given location
    """
    def __init__(self):
        super().__init__(self._generate_message())

    def _generate_message(self):
        return (
            "No gauges are available for the requested location."
            "This usually happens when the either the location was entered with a "
            "typo, when the location does not have a valid country code in the "
            "data/country_code_conversions/ folder, when the location was entered "
            "with a typo, or when the location is not available at all in the API."
        )
    

class GaugeModelsNotAvailableError(Exception):
    """
    Raised when no gauge models are available for the given location
    """
    def __init__(self):
        super().__init__(self._generate_message())

    def _generate_message(self):
        return (
            "No gauge models are available for the requested location."
        )


class ForecastsNotAvailableError(Exception):
    """
    Raised when forecasts are not available for the given location.
    Usually, this happens when the requested time delta is (partially)
    unavailable for the requested time delta, where the time delta is
    the first issue time plus the final issue time plus the lead time
    """
    def __init__(self):
        super().__init__(self._generate_message())

    def _generate_message(self):
        return (
             "Forecasts are not available for the requested location and delta.\n\n"
             "This happens when the requested time delta is (partially) "
             "unavailable for the requested time delta, where the time delta is "
             "(b - a) + the lead time. "
             "Most often, this is caused by the requested issue time being "
             "just too far back into the past (with as limit July 2024 as of writing "
             "this in October 2024)."
        )
    

def handle_exception(exc: Exception) -> None:
    """
    Handle exceptions by printing the error message

    :param exc: the exception that was raised
    """
    if isinstance(exc, requests.HTTPError):
        print(f"HTTPError: {exc}")
    elif isinstance(exc, json.JSONDecodeError):
        print(f"JSONDecodeError: {exc}")
    elif isinstance(exc, GaugesNotAvailableError):
        print(f"GaugesNotAvailableError: {exc}")
    elif isinstance(exc, GaugeModelsNotAvailableError):
        print(f"GaugeModelsNotAvailableError: {exc}")
    elif isinstance(exc, ForecastsNotAvailableError):
        print(f"ForecastsNotAvailableError: {exc}")
    else:
        print(f"An unexpected error occurred: {exc}")