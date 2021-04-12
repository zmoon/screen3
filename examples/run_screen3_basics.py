# -*- coding: utf-8 -*-
"""
Demonstration of basic usage of the Python module `screen3`

You can use `help({fn})` to see the help for each function, i.e., try running these:
    help(screen3)
    help(screen3.run)
    help(screen3.plot)
    help(screen3.read)

@author: zmoon
"""
import matplotlib.pyplot as plt
import numpy as np

import screen3

plt.close('all')


# %% Set exe location
# This may be necessary, e.g.,
# if you have more than one executable in or below your current directory that matches the SCREEN*.exe pattern.
# If SCREEN3.exe is in the same directory as this script, you shouldn't have to set it.

# screen3.set_screen_exe_loc('path/to/an/executable.exe')


# %% Run with default settings

df = screen3.run()

fig = screen3.plot(df)

plt.title("A title")  # we can add a title to the figure after it has been created
plt.tight_layout()

# fig.savefig('fig_our-default-settings.pdf')  # modify the save name here if you want


# %% Demonstrate how to modify input parameters

df = screen3.run(
    RUN_NAME='the best run.',
    Q=500.0,
    HS=20.0,
    DS=3.0,
    TS=300.0,
    VS=10.0,
    ZR=1.0,
    X=np.linspace(1.0, 350.0, 50),
    IMETEO=3,
    ISTAB=4,
    WS=2.0,
    U_or_R='R',
    DOWNWASH_YN='N',
    HB=40.0
)

fig = screen3.plot(df)

# fig.savefig('fig_sample-modified-settings.pdf')


# %% Demonstrate loading saved (in the SCREEN3 output format) past runs

df1 = screen3.load_example('EXAMPLE.OUT')
df2 = screen3.load_example('EXAMPNR.OUT')

fig1 = screen3.plot(df1)
fig2 = screen3.plot(df2)

# fig1.savefig('fig_example.pdf')
# fig2.savefig('fig_exampnr.pdf')
