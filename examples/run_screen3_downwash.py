# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:19:36 2020

Examine impact of building dimensions. 

@author: zmoon
"""

import matplotlib.pyplot as plt
import numpy as np

import screen3

plt.close('all')


#%% establish conditions to test

#> choose a different set if taking too long... 

#> this one takes not long at all
# HB_vals = np.arange(10.0, 50.0, 5.0)  # height
# HW_vals = np.arange(5.0, 100.0, 10.0)  # max horiz dim

# HB_vals = np.arange(10.0, 50.0, 2.0)  # height
# HW_vals = np.arange(5.0, 100.0, 5.0)  # max horiz dim

#> this one takes pretty long (1520 runs!)
HB_vals = np.arange(10.0, 50.0, 1.0)  # height
HW_vals = np.arange(5.0, 100.0, 2.5)  # max horiz dim


X = np.linspace(10.0, 500.0, 50)


#%% run sims

all_runs = []

for hb in HB_vals:
    
    for hw in HW_vals:
    
        hl = 0.7 * hw  # minimum horiz dim as fraction of max
    
        df = screen3.run_screen(HB=hb, HL=hl, HW=hw, DOWNWASH_YN='Y', U_or_R='U', X=X)
        
        all_runs.append((hb, hw, df))
        
        
        
#%% examine impact of building height for one max horiz dim

HW_choice = 45.0

iHW = np.argwhere(HW_vals == HW_choice)[0]


# determine how many to use so we don't have too many in the plot
n_desired = 10
n_tot = HB_vals.size
n_stride = np.floor(n_tot/n_desired).astype(int)

# filter runs
runs = [ run for run in all_runs if run[1] == HW_choice ][::n_stride]

dfs = [ run[2] for run in runs ]

labels = [f"{run[0]:.2g} m" for run in runs ]


fig, ax = plt.subplots()

screen3.plot_conc(dfs, labels=labels, ax=ax)

ax.set_title('Impact of building height (HB)')
ax.set_title(f"HW = {HW_choice}", loc='left', fontsize='small')

fig.tight_layout()

fig.savefig('fig_downwash_HB.pdf')


#%% examine impact of max horiz dim for one building height 

HB_choice = 35.0

iHB = np.argwhere(HB_vals == HB_choice)[0]

# determine how many to use so we don't have too many in the plot
n_desired = 10
n_tot = HB_vals.size
n_stride = np.floor(n_tot/n_desired).astype(int)

# filter runs
runs = [ run for run in all_runs if run[0] == HB_choice ][::n_stride]

dfs = [ run[2] for run in runs ]

labels = [f"{run[1]:.2g} m" for run in runs ]


fig, ax = plt.subplots()

screen3.plot_conc(dfs, labels=labels, ax=ax)

ax.set_title('Impact of building max horiz dimension (HW)')
ax.set_title(f"HB = {HB_choice}", loc='left', fontsize='small')

fig.tight_layout()

fig.savefig('fig_downwash_HW.pdf')


#%% examine both params for one positoin

# X_choices = [150.0]
X_choices = [50.0, 100.0, 150.0, 200.0, 300.0]

w = len(X_choices) * 3
# fig, axs = plt.subplots(1, len(X_choices), figsize=(w, 4), sharex=True, share=True)
# ^ for combined fig

for c, X_choice in enumerate(X_choices):

    # ax = axs.flat[i]

    iX = np.argwhere(dfs[0]['DIST'].values == X_choice)[0]
    
    data = np.empty((HB_vals.size, HW_vals.size))
    
    for i, hb in enumerate(HB_vals):
        
        for j, hw in enumerate(HW_vals):
    
            ind = i*HW_vals.size + j
    
            run = all_runs[ind]
            
            assert run[0] == hb and run[1] == hw
            
            df = run[2]
            
            data[i,j] = df['CONC'].values[iX]
    
    fig, ax = plt.subplots()

    # im = ax.contourf(HW_vals, HB_vals, data, 20)    
    im = ax.pcolormesh(HW_vals, HB_vals, data)
    
    cb = fig.colorbar(im)
    cb.set_label(f'({df.attrs["units"]["CONC"]})')
    # TODO: ^ need to remove attrs thing if don't add it!!
    
    ax.set_ylabel('HB (m)')
    ax.set_xlabel('HW (m)')
    ax.set_title(f'CONC at {X_choice:.3g} m')
    
    fig.savefig(f"fig_downwash_HB-HW_{c+1}.pdf")


