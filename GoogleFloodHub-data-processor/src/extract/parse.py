# src/extract/parse.py

import sys
from typing import Tuple
from datetime import datetime


def parse_args(args: list) -> Tuple:
    """
    Parse the command line arguments,
    If not four (args[0] is script name),
    clarify requested usage and exit.

    :param args: command line arguments
    :return: the country, a, and b
    """
    if len(args) != 4:
        print('Invalid number of arguments, please provide country, start date, and end date')
        print('Usage: python3 main.py <country> <a> <b>')
        print('Example: python3 main.py Mali 01-10-2024 10-10-2024')
        sys.exit(1)

    return args[1], args[2], args[3]


def validate_date(date: str) -> datetime:
    """
    Validate the date format, if not in the format
    of DD-MM-YYYY, clarify requested usage and exit

    :param date: date to validate
    :return: the date in datetime format
    """
    try:
        return datetime.strptime(date, "%d-%m-%Y")
    except ValueError:
        print(f"Error: Invalid date format '{date}', expected format is DD-MM-YYYY")
        sys.exit(1)


def validate_dates(a: datetime, b: datetime) -> None:
    """
    Validate the date order, if a is after b,
    clarify requested usage and exit

    :param a: first date
    :param b: second date
    """
    if a > b:
        print(f"Error: Start date '{a}' is after end date '{b}'")
        sys.exit(1)


def validate_country(country: str) -> str:
    """
    Validate the country format, if not starting with
    a capital letter and containing only alphabetic
    characters, clarify requested usage and exit
    
    :param country: country string to validate
    :return: the country string
    """
    if not country[0].isupper() or not country.isalpha():
        print(
            f"Error: Country '{country}' must start with a capital "
            "letter and contain only alphabetic characters"
        )
        sys.exit(1)
    return country


def validate_args(args: list) -> Tuple:
    """
    Validate the command line arguments using helper functions
    
    :param args: command line arguments
    :return: the country, a (first issue date), and b (final issue date)
    """
    country, a, b = parse_args(args)
    a = validate_date(a)
    b = validate_date(b)
    validate_dates(a, b)
    country = validate_country(country)
    
    return country, a, b