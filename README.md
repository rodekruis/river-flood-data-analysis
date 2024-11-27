# Analysis and comparison of GloFAS and Google FloodHub river discharge data

This repository contains scripts that together perform an analysis of [GloFAS](https://global-flood.emergency.copernicus.eu/)'s and [Google FloodHub](https://sites.research.google/floods/l/0/0/3)'s predictive performance of river discharge data. For both, the data is downloaded, processed to a uniform type, and, subsequently, their accuracies (e.g. by probability of detection (POD) and false alarm ratio (FAR)) are assessed using impact data.

As of now, late 2024, the scripts focus on floods in Mali, but later, the goal is to generalize to more countries. For questions, contact [elskuipers@rodekruis.nl](mailto:elskuipers@rodekruis.nl) and [toldenburg@rodekruis.nl](mailto:toldenburg@rodekruis.nl).

## Overview

An analysis can be divided into three main parts: (1) preparing the GloFAS data; (2) preparing the FloodHub data; and (3) comparing them. The directories are set-up comparably:
1. **Preparing GloFAS data**: *GloFAS* directory contains scripts to extract GloFAS data through an API, and define flood events that are comparable with impacts defined through impact data or observational data (from hydrological stations)
2. **Preparing FloodHub data**: *GoogleFloodHub* directory contains scripts to query the (beta) FloodHub API and transform the downloaded data into a uniform format suitable for comparison with GloFAS forecasts. A set-up and manual for the downloading part can be found [here](https://github.com/valentijn7/GoogleFloodHub-data-extractor.git). Furthermore, ...
3. **Comparison**: visualizing and comparing outcomes of GloFAS performance to Floodhub

## Data

Multiple data sources were utilised. ...

Impact data is not stored online and can be shared upon requested, either through the contact persona above, or through [pphung@redcross.nl](mailto:pphung@redcross.nl).

## Set-up

...

