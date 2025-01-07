# Analysis and comparison of GloFAS and Google FloodHub river discharge data

This repository contains scripts that together perform an analysis of [GloFAS](https://global-flood.emergency.copernicus.eu/)'s and [Google FloodHub](https://sites.research.google/floods/l/0/0/3)'s predictive performance of river discharge data. For both, the data is downloaded, processed to a uniform type, and, subsequently, their accuracies (e.g. by probability of detection (POD) and false alarm ratio (FAR)) are assessed using both impact data and observational data.

As of now, late 2024, the scripts focus on floods in Mali, but later, the goal is to generalize to more countries. For questions, contact [elskuipers@rodekruis.nl](mailto:elskuipers@rodekruis.nl) and [toldenburg@rodekruis.nl](mailto:toldenburg@rodekruis.nl).

## Overview

An analysis of historical forecasts versus a ground truth, i.e. impact- or observational data, can be divided into different parts: (1) preprocessing forecasts to events; (2) preprocessing ground truth to events; (3) comparing them. Here, "events" flexibly denote periods of consecutive flooding in a predefined spatial area. Something is considered a flood when either, in case of quantitative discharge data, a certain return period threshold or percentile treshold is passed, or, in case of qualitative event data, according to the source's flood interpretation.

The repository contains four main folders:
1. **comparison**: contains scripts to directly compare results of different forecasting methods, including plotting, et cetera.
2. **GloFAS**: contains scripts to preprocess GloFAS forecasts into events, and also scripts to preprocess the impact and observational data to these same uniform events.
3. **GoogleFloodHub**: contains scripts to download both real-time forecasts through an API, and download historical forecasts (the GRRR dataset) through an online environment. The latter are transformed to events together with impact- and observational data in the ``GRRR.ipynb`` file.
4. **PTM**: contains scripts to analyse the status quo; the so-called "propagation trigger model" (PTM).

## Data

Multiple data sources were utilised. ...

Impact data is not stored online and can be shared upon requested, either through the contact persona above, or through [pphung@redcross.nl](mailto:pphung@redcross.nl).

## Set-up

...

