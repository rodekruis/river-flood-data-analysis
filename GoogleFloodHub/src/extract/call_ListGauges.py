# src/extract/call_ListGauges.py

from .getters import get_API_key, get_json_file
from .exceptions import GaugesNotAvailableError

from typing import List, Dict, Any
import json
import requests
import pandas as pd


def generate_url_ListGauges(path_to_key : str) -> str:
    """
    Generate the URL to get the list of gauges for a specific country

    :param path_to_key: path to the .txt file containing the API key
    :return: the URL
    """
    base_url = 'https://floodforecasting.googleapis.com/v1/gauges:searchGaugesByArea'
    return f'{base_url}?key={get_API_key(path_to_key)}'


def make_request_ListGauges(
        country_code: str, path_to_key: str, quality_verified: bool = True
    ) -> requests.Response:
    """
    Make the API request for the list of gauges for a specific country

    :param url: the URL
    :param path_to_key: location of the key
    :param quality_verified: whether to download (un)verified gauges
    :return: the list of gauges
    """
    if type(quality_verified) is not bool:
        raise ValueError('quality_verified must be a boolean')
    else:
        qv = not quality_verified # negationary flip
    
    response = requests.post(
        generate_url_ListGauges(path_to_key),
        json = {
            'regionCode': country_code,
            'includeNonQualityVerified': qv
        }
    )
    
    return response


def verify_ListGauges(response : Any) -> Any:
    """
    Verify the ListGauges API call response

    :param response: the response
    :return: the list of gauges
    """
    if response.status_code != 200:
        raise requests.HTTPError(f'Error: {response.status_code} -- {response.text}')

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f'Error parsing JSON: {exc.msg}', exc.doc, exc.pos)
    
    if 'gauges' in data:
        print('ListGauges API call successful')
        return data['gauges']
    else:
        print('Error: no gauges found in the response')
        print('Full response:', data)
        raise GaugesNotAvailableError()
    

def convert_ListGauges_to_df(gauges : List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the list of gauges to a pd.DataFrame

    :param gauges: the list of gauges
    :return: the DataFrame
    """
    df = pd.DataFrame(gauges)
    df['latitude'] = df['location'].apply(lambda x: x['latitude'])
    df['longitude'] = df['location'].apply(lambda x: x['longitude'])
    df.drop(columns = ['location'], inplace = True)
    # sort by gaugeId to ensure same response result for every API call;
    # this also makes the orger consistent between ListGauges and GetGaugeModel
    df['gaugeId'] = df['gaugeId'].astype(str).str.strip()
    df = df.sort_values(by = 'gaugeId').reset_index(drop = True)

    return df


def get_ListGauges(country : str, path_to_key : str, quality_verified: bool = False) -> pd.DataFrame:
    """
    Get the list of gauges for a specific country by calling helper functions which:
    - generate the URL;
    - make the request;
    - verify the response; and
    - convert to a pd.DataFrame in the end.

    :param country: the country
    :param path_to_key: path to the .txt file containing the API key
    :param quality_verified: whether to download (un)verified gauges. The default is False,
                             but it gets negated later on. This is confusing, but the other
                             function calls work the other way around...
    :return: DataFrame with the gauges
    """
    return convert_ListGauges_to_df(
        verify_ListGauges(
            make_request_ListGauges(
                get_json_file("../data/country_code_conversions/country_codes.json")[country],
                path_to_key,
                quality_verified
            )
        )
    )