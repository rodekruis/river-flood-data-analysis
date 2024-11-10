# src/extract/call_QueryGaugeForecasts.py

from .getters import get_API_key
from .exceptions import ForecastsNotAvailableError

from typing import List, Dict, Any
import json
import requests
import datetime
import pandas as pd


def make_request_QueryGaugeForecasts(
        path_API_key: str, gauge_ids: List[str], interval: tuple) -> Any:
    """
    Make the QueryGaugeForecasts API call
    
    :param path_API_key: path to the file containing the API key
    :param gauge_ids: list of gauge IDs
    :param interval: tuple of two datetime objects
    :return response: response from the API call
    """
    return requests.get(
            f'https://floodforecasting.googleapis.com/v1/gauges:queryGaugeForecasts',
            params = {
                'key': get_API_key(path_API_key),
                'gaugeIds': gauge_ids,
                'issuedTimeStart': interval[0].strftime("%Y-%m-%d"),
                'issuedTimeEnd': interval[1].strftime("%Y-%m-%d"),
            },
        )


def verify_response_QueryGaugeForecasts(response: Any) -> Any:
    """
    Verify the response from the QueryGaugeForecasts API call
    
    :param response: response from the API call
    :return: True if the response is valid, False otherwise
    """
    if response.status_code != 200:
        raise requests.HTTPError(f'Error: {response.status_code} -- {response.text}')

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f'Error parsing JSON: {exc.msg}', exc.doc, exc.pos)
    
    if 'forecasts' in data:
        print('QueryGaugeForecasts API call successful\n')
        return data['forecasts']
    else:
        print('Error: no forecasts found in the response')
        print('Full response:', data, '\n')
        raise ForecastsNotAvailableError()
    

def convert_QueryGaugeForecasts_to_df(response: Dict[str, Dict[str, List[Any]]]) -> List[pd.DataFrame]:
    """
    Convert the response from the QueryGaugeForecasts API call to DataFrame(s).
    For now, we format the data into a Long format, as it allows for implicit
    multidimesnionality and pandas-optimised operations
    
    :param response: response from the API call
    :return: DataFrame
    """
    records = []
    # flatten the data by de-nesting the JSON response
    for gauge_id, gauge_data in response.items():

        for forecast in gauge_data['forecasts']:
            issued_time = forecast['issuedTime']

            for forecast_data in forecast['forecastRanges']:
                # slice the date string to only include the date
                forecast_date = forecast_data['forecastStartTime'][:10]
                forecast_value = forecast_data['value']
                records.append({
                    'gaugeId': gauge_id,
                    'issue_time': issued_time,
                    'fc_date': forecast_date,
                    'fc_value': forecast_value,
                })

    df = pd.DataFrame(records)
    df['issue_time'] = pd.to_datetime(df['issue_time'])
    df['issue_date'] = df['issue_time'].dt.date
    df['fc_date'] = pd.to_datetime(df['fc_date'])        
    df = df[['gaugeId', 'issue_date', 'issue_time', 'fc_date', 'fc_value']]

    return df


def get_QueryGaugeForecasts(
        path_API_key: str, gauge_ids: List[str], interval: tuple) -> List[pd.DataFrame]:
    """
    Get the QueryGaugeForecasts API call, which returns:
    - for every gauge in gauge_ids, the forecasts for the given interval, which contain:
        - the gauge ID
        - the issued time
        - the forecasted dates (lead time is (usually a week))
        - the forecasted values (in cubic meters per second)
        
    (Other relevant metadata can be queried per gauge using the gauge ID
    through the GetGaugeModel API call, function get_GetGaugeModel())
    
    :param path_API_key: path to the file containing the API key
    :param gauge_ids: list of gauge IDs
    :param interval: tuple of two datetime objects
    :return: DataFrame
    """
    return convert_QueryGaugeForecasts_to_df(
        verify_response_QueryGaugeForecasts(
            make_request_QueryGaugeForecasts(path_API_key, gauge_ids, interval)
        )
    )