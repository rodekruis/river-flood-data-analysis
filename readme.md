# Analysis and comparison of GloFAS and Google FloodHub river discharge data

This repository contains scripts that together perform an analysis of [GloFAS](https://global-flood.emergency.copernicus.eu/)'s and [Google FloodHub](https://sites.research.google/floods/l/0/0/3)'s predictive performance of river discharge data. For both, the data is downloaded and transformed, then processed to a uniform type, and then their accuracies (e.g. by probability of detection (POD) and false alarm ratio (FAR)) are assessed using impact data.

As of now, late 2024, the scripts focus on floods in Mali, but later, the goal is to generalize to more countries. For questions about GloFAS, contact [elskuipers@rodekruis.nl](mailto:elskuipers@rodekruis.nl), and for questions about FloodHub, contact [toldenburg@rodekruis.nl](mailto:toldenburg@rodekruis.nl).

## Overview

An analysis can be divided into three main parts: (1) preparing the GloFAS data; (2) preparing the FloodHub data; and (3) comparing them. The directories are set-up comparably:
1. **Preparing GloFAS data**: ...
2. **Preparing FloodHub data**: the 'GoogleFloodHub-data-extractor' directory contains code to query the (beta) FloodHub API, of which the set-up and manual can be found [here](https://github.com/valentijn7/GoogleFloodHub-data-extractor.git). 'GoogleFloodHub-data-processor' transforms the downloaded data in a uniform format suitable for comparison with GloFAS forecasts.
3. **Comparison**: ...


## Set-up

...

