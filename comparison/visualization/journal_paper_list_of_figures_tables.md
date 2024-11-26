- [ ] Figure 1: initial data + study area map combo
    - [ ] Raw glofas raster data for one day
    - [ ] administrative unit overlay (black lines?) + names 
    - [ ] Google Floodhub stations
    - [ ] DNH hydrological stations

- [ ] Table 1: Contingency table 

- [ ] Table 2: Performance measures 

- [ ] Table 3: Configuration settings (see configuration.py)

- [ ] Figure 2: GloFAS performance, 4 maps of Mali divided by administrative units, for settings: RP=5yr , leadtime=7 days: 
    - [ ] FAR for obs & imp
    - [ ] POD for obs & imp

- [ ] Figure 3: Google floodhub performance,  4 maps of Mali, divided by administrative units, for settings RP=5yr , leadtime=7 days 
    - [ ] FAR for obs & imp
    - [ ] POD for obs & imp

- [ ] Figure 4: for one administrative unit (pick one with high risk + many datapoints), containing 4 plots
 each plot with lines for values of each model [GloFAS, Google Floodhub, EAP] against each metric [observation,     impact] = 6 lines (3 colours, 2 linestyles)
    - [ ] plot 1: POD against leadtime
        - x-axis : leadtime (hours)
        - y-axis : POD (0-1, no unit)
    - [ ] plot 2: POD against return period 
        - x-axis: return period (years)
        - y-axis: POd (-) 
    - [ ] plot 3: FAR against leadtime 
        - x-axis: leadtime (hr)
        - y-axis: FAR (-) 
    - [ ] plot 4: FAR against return period threshold
        - x-axis: return period (yr) 
        - y-axis: FAR (-) 

- [ ] Figure 5: performance of models GloFAS AND Floodhub / GloFAS OR Floodhub , depending which one performed best