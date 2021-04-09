# -*- coding: utf-8 -*-
"""
Examine the response to meteorological conditions
* stability class
* mean wind speed
by plotting the conc lines. 

@author: zmoon
"""
import matplotlib.pyplot as plt
import numpy as np

import screen3

plt.close('all')


# %% Define conditions
# https://en.wikipedia.org/wiki/Outline_of_air_pollution_dispersion#The_Pasquill_atmospheric_stability_classes


# SCREEN specifies maxima: UINMAX/3.,5.,10.,20.,5.,4./ (L2874)
# that we must obey (or the run will not complete)
stab_classes = {
    # description: (allowed WS range, ISTAB for SCREEN3)
    'A': ('very unstable', (0, 3), 1),  # 
    'B': ('unstable', (0, 5), 2), 
    'C': ('slightly unstable', (2, 10), 3),  # max WS should be None
    'D': ('neutral', (3, 20), 4),  # max WS should be None
    'E': ('slightly stable', (0, 5), 5),
    'F': ('stable', (0, 3), 6)
}


#%% Loop through conditions and run SCREEN

X = np.linspace(10.0, 1000.0, 50)
IMETEO = 3  # one ISTAB, one WS 
n_ws_per_stab = 10
wsmin = 1.0  # SCREEN won't run for values below this

# Get ready to store stuff
runs = []

for stab, d in stab_classes.items():
    ws_range = d[1]
    ISTAB = d[2]
    
    # Create list of WS values to run for
    ws_a = max(float(ws_range[0]), wsmin)
    ws_b = float(ws_range[1]) if ws_range[1] is not None else 20.0

    ws_vals = np.linspace(ws_a, ws_b, n_ws_per_stab)

    dfs = []
    for ws_val in ws_vals:
        df = screen3.run(X=X, IMETEO=3, ISTAB=ISTAB, WS=ws_val)
        dfs.append(df)
        
    # Save our stuff
    runs.append((stab, dfs, ws_vals))
    

# %% Plot on one figure

fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True)

for (stab, dfs, wss), ax in zip(runs, axs.flat):
    
    labels = [f"{ws:.2g} m/s" for ws in wss]
    
    screen3.plot(dfs, labels=labels, ax=ax)
    
    ax.set_title(f"{stab} ({stab_classes[stab][0]})")


# fig.savefig('fig_meteo-lines.pdf')
