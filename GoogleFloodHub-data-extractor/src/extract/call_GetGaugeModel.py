# src/extract/call_GetGaugeModel.py

from .getters import get_API_key
from .exceptions import GaugeModelsNotAvailableError

from typing import List, Dict, Any
import json
import requests
import numpy as np
import pandas as pd


def generate_model_names(df_gauges: pd.DataFrame) -> List[str]:
    """
    Generate the model names for the GetGaugeModel API call

    :param df_gauges: a DataFrame containing the gauge IDs
    :return: a list of strings, each of which is a model name
    """
    return [f'names=gaugeModels/{id}' for id in df_gauges['gaugeId'].tolist()]


def generate_url_GetGaugeModel(path_to_key: str, df_gauges: pd.DataFrame) -> str:
    """
    Generate the URL for the GetGaugeModel API call

    :param path_to_key: the path to the API key
    :param df_gauges: a DataFrame containing the gauge IDs
    :return: the URL
    """
    base_url = 'https://floodforecasting.googleapis.com/v1/gaugeModels:batchGet'
    model_names_parameter = '&'.join(generate_model_names(df_gauges))
    return f'{base_url}?key={get_API_key(path_to_key)}&{model_names_parameter}'


def make_request_GetGaugeModel(path_to_key: str, df_gauges: pd.DataFrame) -> requests.Response:
    """
    Make the GetGaugeModel API call and return the response as a dictionary
    
    :param path_to_key: the path to the API key
    :param df_gauges: a DataFrame containing the gauge IDs
    :return: a dictionary containing the response
    """
    response = requests.get(
        generate_url_GetGaugeModel(path_to_key, df_gauges)
    )

    return response


def verify_GetGaugeModel(response: Any) -> Any:
    """
    Verify that the GetGaugeModel API call is working correctly

    :param path_to_key: the path to the API key
    :param df_gauges: a DataFrame containing the gauge IDs
    :return: a dictionary containing the response
    """
    if response.status_code != 200:
        raise requests.HTTPError(f'Error: {response.status_code} -- {response.text}')

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f'Error parsing JSON: {exc.msg}', exc.doc, exc.pos)
    
    if 'gaugeModels' in data:
        print('GetGaugeModel API call successful')
        return data['gaugeModels']
    else:
        print('Error: no gaugeModels found in the response')
        print('Full response:', data)
        raise GaugeModelsNotAvailableError()


def convert_GetGaugeModel_to_df(response: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the response from the GetGaugeModel API call to a DataFrame

    :param response: the response from the GetGaugeModel API call
    :return: a DataFrame containing the gauge models
    """
    df = pd.DataFrame(response)

    # As for some stations, a threshold value is not available, we need to check if
    # the key exists in the df before trying to access it (and getting a KeyError):
    # This works because .get() automatically returns None if the key does not exist
    df['warningLevel'] = df['thresholds'].apply(lambda x: x.get('warningLevel') \
                                                if isinstance(x, dict) else np.nan)
    df['dangerLevel'] = df['thresholds'].apply(lambda x: x.get('dangerLevel') \
                                               if isinstance(x, dict) else np.nan)
    df['extremeDangerLevel'] = df['thresholds'].apply(lambda x: x.get('extremeDangerLevel') \
                                                      if isinstance(x, dict) else np.nan)

    # # Retrieve all threshold values from Dict in column entries using lambda ()
    # df['dangerLevel'] = df['thresholds'].apply(lambda x: x['dangerLevel'])
    # df['extremeDangerLevel'] = df['thresholds'].apply(lambda x: x['extremeDangerLevel'])
    # df['warningLevel'] = df['thresholds'].apply(lambda x: x['warningLevel'])
    df.drop(columns = ['thresholds'], inplace = True)

    return df


def get_GetGaugeModel(path_to_key: str, df_gauges: pd.DataFrame) -> pd.DataFrame:
    """
    Get the gauge models for a list of gauge IDs, including information about (in order):
    - gauge ID again
    - the gauge value unit (usually cubic meters per second)
    - whether the gauge quality is verified
    - thresholds:
        - danger level
        - extreme danger level
        - warning level

    These will be returned as a pd.DataFrame with the thresholds in unique columns

    :param path_to_key: the path to the API key
    :param df_gauges: a DataFrame containing the gauge IDs
    :return: a DataFrame containing the gauge models
    """
    return convert_GetGaugeModel_to_df(
        verify_GetGaugeModel(
            make_request_GetGaugeModel(path_to_key, df_gauges)
        )
    )