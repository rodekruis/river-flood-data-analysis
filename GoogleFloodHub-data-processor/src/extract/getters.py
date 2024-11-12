# src/extract/getters.py

from typing import Dict
import json


def get_API_key(path : str) -> str:
    """
    Get the API key from a .txt file stored elsewhere

    :param path: path to the .txt file containing the API key
    :return: the API key
    """
    with open(path, 'r') as f:
        return f.read().strip()


def get_json_file(path : str) -> Dict:
    """
    Get .json file stored elsewhere

    :param path: path to the .json file
    :return: the file as a dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)