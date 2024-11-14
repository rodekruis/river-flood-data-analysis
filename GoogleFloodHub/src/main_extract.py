# src/main_extract.py

# main() to run the extract package from the command line.
# As input, give:
# - country: the country name of interest;
# - starting date: the first issue date of interest; and
# - ending date: the final issue date of interest.
# Example usage: python3 main.py Mali 01-10-2024 07-10-2024
#
# For a more complete instruction, see the README of:
# https://github.com/valentijn7/GoogleFloodHub-data-extractor;
# For the most recent version of the code, see:
# https://github.com/rodekruis/river-flood-data-analysis


import sys
import extract


def main():
    print("BEWARE: as of November 2024, the API contains no data issued before July 2024\n")
    
    try:
        country, a, b = extract.validate_args(sys.argv)
        _, _, forecasts = extract.extract_country_data_for_time_delta("../key.txt",
                                                                      country,
                                                                      (a, b))
        extract.validate_forecasts(forecasts, (a, b), country)
    except Exception as exc:
        extract.handle_exception(exc)


if __name__ == '__main__':
    main()