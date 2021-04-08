# -*- coding: utf-8 -*-
"""
Python functions for using SCREEN3
* wrapper to quickly create a run without needing to go through the prompt
* load the output into Pandas
* some plotting routines

Code in here should generally not need to be modified for basic runs. 

Python 3.6+ is required

@author: zmoon
"""
# TODO: type annotations for the main user-facing fns?

import datetime
import os
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# check for Python 3.6+
# `sys.version_info` became named tuple first in 3.1
if not (sys.version_info.major >= 3 and sys.version_info.minor >= 6):
    raise Exception(f"Python version must be 3.6+.\nYours is {sys.version}")


# TODO: on Jose's Mac it set to use the Win and failed to to multi-run stuff, but the checks did not activate! 
# (didn't see erorrs on Jose's screen)
# it just ran one case for meteo/downwash and gave no answers


_SCREEN_EXE_PATH = None

def set_screen_exe_loc(fp):
    """Set module variable for path of the SCREEN3.exe.
    
    INPUT
    -----
    fp : str
        path (absolute or relative) to the SCREEN3 executable
        e.g., './screen3_Win/SCREEN3.exe'

    """
    global _SCREEN_EXE_PATH
    p = Path(fp)

    if p.is_file():
        _SCREEN_EXE_PATH = p
    else:
        raise ValueError(f"The path {fp!r} does not exist or is not a file.")


# TODO: write this fn that will try the example as a check 
#       on module load?
# def _try_exe_example():
#         try:
#         completed = subprocess.run(
#             # f'SCREEN3.exe < {cwd}/{ifn}',
#             ['SCREEN3.exe', '<', f'{ifp}'],
#             # shell=True,
#             shell=False,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             # check=True,
#         )
#         # print(completed)
#         run_complete = True

#     # except subprocess.CalledProcessError as err:  # for use with shell=True, check=True
#     #     print(err)

#     except OSError:
#         print(f"The executable {_SCREEN_EXE_PATH} is not compatible with your platform:"
#             f"\n  {platform.platform()} of general type {platform.system()}")
#     except:
#         import sys
#         err = sys.exc_info()[0]
#         print("Unexpected error:")
#         print(err)
#     else:
#         print("wat")


def _try_to_set_screen_exe_loc():
    set_msg = "use `screen3.set_screen_exe_loc` to set location of the SCREEN3 executable to use."
    if _SCREEN_EXE_PATH is None:
        # first try standard loc
        try:
            set_screen_exe_loc('./SCREEN3.exe')
        except ValueError:
            print("the executable was not found in the expected location './'"
                "\ni.e., './SCREEN3.exe' not found"
                )
            print("will now search for one")
            # search for an exe
            exe_paths = list(Path.cwd().rglob('SCREEN*.exe'))
            if len(exe_paths) > 1:
                print("multiple executables found:")
                print('\n'.join([f"    {sp.relative_to(Path.cwd())}"  for sp in exe_paths]))
                print(set_msg)
            elif len(exe_paths) == 1:
                set_screen_exe_loc(str(exe_paths[0]))
            else:
                print("couldn't find any SCREEN exe files.")
                print(set_msg)

_try_to_set_screen_exe_loc()


# note that U10M becomes UHANE in non-regulatory mode
SCREEN_OUT_COL_NAMES = 'DIST CONC STAB U10M USTK MIX_HT PLUME_HT SIGMA_Y SIGMA_Z DWASH'.split()
SCREEN_OUT_COL_UNITS = ['m', 'μg/m$^3$', '', 'm/s', 'm/s', 'm', 'm', 'm', 'm', '']
SCREEN_OUT_COL_UNITS_DICT = dict(zip(SCREEN_OUT_COL_NAMES, SCREEN_OUT_COL_UNITS))
# ^ maybe _DIST_ instead of _COL_ ?

def read_screen(fp, 
    *, 
    t_run=None,
    run_inputs={},
):
    """Read and extract data from a SCREEN3 run (a SCREEN.OUT file).
    
    INPUTS
    ------
    fp : str
        path to the SCREEN.OUT file

    INPUTS (optional)
    ------
    t_run : 
        time of the run
        if the output file (SCREEN.OUT) is older than this, the run did not complete correctly
    run_inputs : dict
        the run script passes this so that we can store what the inputs to the Python function were for this run
        though SCREEN.OUT should have much of the same info

    RETURNS
    -------
    df : pd.DataFrame
        of the data
        .attrs: 
            units : dict
                keys: the variables names
                values: unit strings (for use in plots)
            SCREEN_OUT : str
    """
    # find where table starts (in case it isn't always in the same place)
    # with open(fp, 'r') as f:
    #     for i, line in enumerate(f):
    #         if line.split()[:2] == ['DIST', 'CONC']:
    #             iheader = i
    #         elif line.strip()[:5] == 'DWASH':
    #             iDWASH = i
    #             break
    # TODO: if they are not found we get UnboundLocalError in this fn
    # should do this a better way so that we can report that SCREEN failed if they are not found.

    # first check that the file has indeed been modified
    p = Path(fp)
    t_out_modified = datetime.datetime.fromtimestamp(p.stat().st_mtime)
    # print(t_run)
    # print(t_out_modified)
    if t_run is not None and p.is_file():  # i.e., t_run has been specified
        # print('checking times')
        if t_out_modified < t_run:  
        # if abs(t_out_modified - t_run) > datetime.timedelta(seconds=1):  # for Kayla, who on her MacBook was having precision erorr. the t_out_modified doesn't have the fraction seconds, so it thought it was earlier...
            raise ValueError(f"SCREEN.OUT is older than time of the run."
                " This probably means that SCREEN failed to run because it didn't like the value of one or more inputs."
                " Or some other reason.")

    with open(fp, 'r') as f:
        lines_raw = f.readlines()

    lines_stripped = [line.strip() for line in lines_raw]

    # TODO: could also read other info from the file like the run settings at the top

    iheader = [i for i, s in enumerate(lines_stripped) if s.split()[:2] == ['DIST', 'CONC']]

    if not iheader:
        raise ValueError(f"Data table not found in output."
            " This probably means that SCREEN failed to complete the run because it didn't like the value of one or more inputs."
            " Or some other reason.")

    if len(iheader) > 1:
        print(f"Multiple sets of data found. Only loading the first one.")
    iheader = iheader[0]

    iblankrel = lines_stripped[iheader:].index("")  # find first blank after iheader
    iblank = iheader + iblankrel


    # could (technically) read these from file
    col_names = SCREEN_OUT_COL_NAMES
    col_units = SCREEN_OUT_COL_UNITS
    
    n_X = iblank - iheader - 2

    # at some point `pd.read_table` was deprecated but seems like that was undone?
    # load into pandas
    df = pd.read_table(fp, 
        sep=None, skipinitialspace=True, engine='python',
        keep_default_na=False,
        skiprows=iheader+2, nrows=n_X)
    df.columns = col_names
    
    units_dict = dict(zip(col_names, col_units))

    # p_dat = p.parents[0] / 'SCREEN.DAT'
    p_dat = p.parents[0] / f'{p.stem}.DAT'
    with open(p_dat, 'r') as f:
        s_screen_dat = f.read()

    #> assign attrs to df
    #  Pandas only added attrs recently. they may not have
    # df.attrs.update({
    #     'SCREEN_DAT': s_screen_dat,
    #     'SCREEN_OUT': ''.join(lines_raw),
    #     'units': units_dict,
    #     'run_inputs': run_inputs
    # })
    
    return df


def run_screen(
    *,
    RUN_NAME='A point-source SCREEN3 run',
    Q=100.0,
    HS=30.0,
    DS=10.0,
    VS=2.0,
    TS=310.0,
    TA=293.0,
    ZR=1.0,
    #
    X=np.append(np.r_[1.0], np.arange(50.0, 500.0+50.0, 50.0)),
    #
    IMETEO=3,
    ISTAB=4,
    WS=5.0,  # there is a max!
    #
    U_or_R='R',
    #
    DOWNWASH_YN='N',
    HB=30.0,  # BUILDING HEIGHT (M)
    HL=10.0,  # MINIMUM HORIZ. BUILDING DIMENSION (M)
    HW=20.0,  # MAXIMUM HORIZ. BUILDING DIMENSION (M)
    #
):
    """Create SCREEN3 input file, feed it to the executable, load the result.

    Inputs must be specified as keyword arguments, but can be entered in any order (non-positional).

    INPUTS
    ------
    Q : float 
        EMISSION RATE (G/S)
    HS : float 
        STACK HT (M)
    DS : float 
        STACK INSIDE DIAMETER (M)
    VS : float 
        STACK GAS EXIT VELOCITY (M/S)
    TS : float 
        STACK GAS TEMPERATURE (K)
    TA : float 
        AMBIENT AIR TEMPERATURE (K)
    ZR : float 
        RECEPTOR HEIGHT ABOVE GROUND (FLAGPOLE RECEPTOR) (M)
    X : array_like
        array of downwind distances (m)
    IMETEO : int {1, 2, 3}
        1 - full (supplying stability class + WS for each grid point), 
            or SCREEN runs for each STAB option??...
        2 - single stability class, 
        3 - stability class + WS
    ISTAB : int
        1(=A) to 6(=F)
    WS : float
        mean wind speed at 10m (m/s)
    U_or_R : str {'U', 'R'}
        urban (U) or rural (R)
    DOWNWASH_YN : str {'Y', 'N'}
        whether to apply building downwash calculations
        the building dimension parameters do nothing if it is 'N'
    HB : float
        BUILDING HEIGHT (M)
    HL : float 
        MINIMUM HORIZ. BUILDING DIMENSION (M)
    HW : float 
        MAXIMUM HORIZ. BUILDING DIMENSION (M)

    RETURNS
    -------
    df : pd.DataFrame
        of the results
        facilitated by function `read_screen`

    NOTES
    -----
    This input data file is created in the current working directory.
    The SCREEN3 program parses it and makes a copy in its directory called 'SCREEN.DAT'. 
    Upon running, it produces an output file in its directory called 'SCREEN.OUT'. 

    """
    inputs = locals()
    # print(inputs)

    # TODO: should check wind speeds

    if _SCREEN_EXE_PATH is None or not isinstance(_SCREEN_EXE_PATH, Path):
        raise ValueError("Before running the location of the executable must be set using `screen3.set_screen_exe_loc`.")

    if not _SCREEN_EXE_PATH.is_file():
        raise ValueError("{fp!r} does not exist or is not a file."
        " Use `screen3.set_screen_exe_loc` to set it.")

    H_defaults = (30.0, 10.0, 20.0)  # remember to change this if change the default arguments above!?
    if any(x0 != x for x0, x in zip(H_defaults, [HB, HL, HW])) and DOWNWASH_YN == 'N':
        print("* Note: You have modified a building parameter, but downwash is not enabled.")

    # cwd = os.getcwd()
    cwd = Path.cwd()

    X = np.asarray(X)
    if X.size > 50:
        raise ValueError('SCREEN3 does not support inputting more than 50 distance values')

    # ------------------------------------------------------
    # for now the user cannot set these
    # though they are required inputs to the model
    
    I_SOURCE_TYPE = 'P'  # point source

    # complex terrain also currently not used
    # (assumed flat)
    TERRAIN_1_YN = 'N'  # complex terrain screen above stack height
    TERRAIN_2_YN = 'N'  # simple terrain above base
    # HTER = 10.  # MAXIMUM TERRAIN HEIGHT ABOVE STACK BASE (M)
    # ^ not needed if not using the terrain settings
    # at least for simple terrain, seems to just run the x positions for different heights 

    # currently user input of distances is required
    # i.e., automated option not used
    AUTOMATED_DIST_YN = 'N'
    DISCRETE_DIST_YN = 'Y'

    # final questions
    FUMIG_YN = 'N'
    PRINT_YN = 'N'

    # ------------------------------------------------------
    # create the input file

    def s_METEO():
        if IMETEO == 1:  # A range (depending on U/R, type of source, etc.) of WS and STABs are examined to find maximum impact
            # print(f"* Note that IMETEO={IMETEO!r} isn't really implemented presently..."
            #     "\nExamine the WS and STAB in the model output to see what happened.")
            return '1'
        elif IMETEO == 2:
            return f'{IMETEO}\n{ISTAB}'
        elif IMETEO == 3:
            return f'{IMETEO}\n{ISTAB}\n{WS}'
        else:
            raise ValueError(f'invalid `IMETEO` {IMETEO!r}. Must be in {{1, 2, 3}}')            

    def s_X():
        vals = X.astype(str).tolist()
        return '\n'.join(vals + ['0.0'])

    def s_building_props():
        if DOWNWASH_YN == 'Y':
            return f'{HB}\n{HL}\n{HW}'
        else:
            return ""

    dat_text = f"""
{RUN_NAME}
{I_SOURCE_TYPE}
{Q}
{HS}
{DS}
{VS}
{TS}
{TA}
{ZR}
{U_or_R}
{DOWNWASH_YN}
{s_building_props()}
{TERRAIN_1_YN}
{TERRAIN_2_YN}
{s_METEO()}
{AUTOMATED_DIST_YN}
{DISCRETE_DIST_YN}
{s_X()}
{FUMIG_YN}
{PRINT_YN}
""".strip()

    ifn = 'screen3_input.txt'  # input filename
    ifp = cwd / ifn

    # TODO: optionally use Python to pass the text stream from memory instead of actually creating the file?
    with open(ifn, 'w') as f: 
        f.write(dat_text)

    # ------------------------------------------------------
    # run the SCREEN executable

    t_utc_run = datetime.datetime.now()

    # if not os.path.isfile(f'{src_dir}/SCREEN3.exe'):
    #     raise ValueError(f'SCREEN3.exe not found in {src_dir}. Please set input param `src_dir` to the proper location.')

    # get an absolute path to the exe
    # the system call will not work on Linux/Mac with a relative path that doesn't start with `./` (pathlib doesn't give that prefix)
    # but it does with if the full path is provided
    exe = str(_SCREEN_EXE_PATH.absolute())

    # src_dir = str(_SCREEN_EXE_PATH.parents[0])
    src_dir = _SCREEN_EXE_PATH.parents[0]

    # move to the src dir because SCREEN creates its files in the place where it is run
    # and we want them to be there
    os.chdir(src_dir)

    # prepare our command for the possibility of absolute paths with spaces in them!
    if platform.system() == 'Windows':
        # cmd = f'""{exe}" < "{ifp}""'  # must enclose whole thing with "" if running like `cmd /C "command"` (or with `os.system()` !)
        cmd = f'"{exe}" < "{ifp}"'  # single quote enclosure won't work for cmd.exe
    elif platform.system() in ('Linux', 'Darwin'):
        cmd = f"'{exe}' < '{ifp}'"
    else:
        raise NotImplementedError

    # print(cmd)

    # os.system('SCREEN3.exe < screen3_input.txt')  # will work on Windows (but not Mac/Linux), if `src_dir` and `cwd` are the same!
    # os.system(cmd)  # for Windows, this requires the version above with the whole command surrounded in double quotes

    subprocess.run(  # returns a CompletedProcess
        # args=[exe, '<', f'{ifp}'],   # in Linux args after the first are passed to the shell instead of exe apparently if list is used with `shell=True`
        # args=f"{exe} < {ifp}",
        args=cmd,
        shell=True,  # * note: need to use list if not `shell=True`
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    # on Win, using `os.system` popped up CMD window, 
    # but with `shell=True` in `subprocess.run` nothing showed in Python interpreter or elsewhere I could see
    # on Linux, stdout was showing until I added the redirection

    # return to working dir
    os.chdir(cwd)

    # ------------------------------------------------------
    # read and parse the output

    # df, _ = read_screen(f'{src_dir}/SCREEN.OUT', t_run=t_utc_run)
    # df = read_screen(f'{src_dir}/SCREEN.OUT', t_run=t_utc_run, run_inputs=inputs)
    df = read_screen(src_dir/'SCREEN.OUT', t_run=t_utc_run, run_inputs=inputs)

    return df




def add_units(x_units, y_units, *, ax=None):
    """Add units and make room."""
    if ax is None:
        ax = plt.gca()
    xl = ax.get_xlabel()
    yl = ax.get_ylabel()
    sxu = f'({x_units})' if x_units else ''
    syu = f'({y_units})' if y_units else ''
    ax.set_xlabel(f'{xl} {sxu}' if xl else sxu)
    ax.set_ylabel(f'{yl} {syu}' if xl else syu)
    # fig = plt.gcf()
    # fig.tight_layout()



def plot_conc(
    df, 
    *, 
    labels=[], 
    yvals=[],
    yvar="",
    yvar_units="",
    plot_type='line', 
    ax=None,
    **pyplot_kwargs):
    """Plot conc.

    The first argument (positional) can be one `df` or a list of `df`s. 
    If you pass a list of `df`s, you can also pass `labels`. 

    The default plot type is 'line', but 2-D plots 'pcolor' and 'contourf' are also options. 
    Unless you additionally pass `yvals`, they will be plotted as if the labels were equally spaced. 

    INPUTS (required)
    ------
    df : pd.DataFrame  or  list of them
        of the data extracted from the SCREEN.OUT of the run

    INPUTS (optional)
    ------
    labels : list (or array)
        labels for the separate cases 
        used if input `df` is a list of output dfs instead of just one
        currently we assume that the first part of the label is the variable name that is being varied
    yvals : list (or array)
        options positions of labels
    yvar : str
        y variable name
        only used if labels not provided
    yvar_units : str
        units to use when labeling yvar
        only used if labels not provided
    plot_type : str {'line', 'pcolor', 'contourf'}
        type of plot to do
    ax : plt.Axes
        if not passed, create a new figure and ax
        else plot on the passed ax

    **pyplot_kwargs
        passed through to the relevant pyplot plotting function

    """
    units = SCREEN_OUT_COL_UNITS_DICT
    # or df.attrs['units']

    if not isinstance(df, (pd.DataFrame, list)):
        raise TypeError(f"df is not a `pd.DataFrame` or `list`, it is type `{type(df).__name__}`")

    if plot_type not in ['line', 'pcolor', 'contourf']:
        raise ValueError(f"`plot_type` {plot_type!r} not allowed")

    if ax is None or not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # determine number of points
    if isinstance(df, list):
        n = df[0].index.size
    else:
        n = df.index.size
    
    # some line styling
    if n > 20:
        line_style = '.-'
        ms = 6
    else:
        line_style = 'o-'
        ms = 5
    pyplot_kwargs.update({
        'markersize': ms
    })

    if isinstance(df, list):
        if not list(labels):
            labels = list(range(1, len(df)+1))
    
        if plot_type == 'line':
            with mpl.rc_context({'legend.fontsize': 'small'}):
                for i, dfi in enumerate(df):
                    dfi.plot(x='DIST', y='CONC', style=line_style, ax=ax, label=labels[i], **pyplot_kwargs) 
        
        else:  # 2-D plot options
            ax.set_xlabel('DIST')
            yvals = np.asarray(yvals)
            if yvals.any():
                Y = yvals

                # yvar_guess = labels[0].split()[0]
                # try:
                #     yvar_units = units[yvar_guess]
                # except KeyError:
                #     raise ValueError(f"Y variable guess {yvar_guess!r} based on labels passed not found in allowed list."
                #         " Make sure that the first part of each label is the variable, followed by a space.")
                # ax.set_ylabel(yvar_guess)

                if labels:
                    ax.set_yticks(yvals)
                    ax.set_yticklabels(labels)

                ax.set_ylabel(yvar)
                add_units(units['DIST'], yvar_units, ax=ax)

            else:
                Y = labels
                add_units(units['DIST'], '', ax=ax)

            X = df[0].DIST
            # Y = yvals if yvals.any() else labels
            Z = np.stack([dfi['CONC'] for dfi in df])

            if plot_type == 'pcolor':
                im = ax.pcolormesh(X, Y, Z, **pyplot_kwargs)

            elif plot_type == 'contourf':
                im = ax.contourf(X, Y, Z, **pyplot_kwargs)
            
            cb = fig.colorbar(im)

            cb.set_label(f"CONC ({units['CONC']})")



    else:  # just one to plot
        if plot_type != 'line':
            raise ValueError("With only one run, only `plot_type` 'line' can be used")
        df.plot(x='DIST', y='CONC', style=line_style, ax=ax, **pyplot_kwargs)

    if plot_type == 'line':
        ax.set_ylabel('CONC')
        add_units(units['DIST'], units['CONC'], ax=ax)  # use the units dict to look up the units

        # ax.set_xlim(xmin=0, xmax=)
        ax.autoscale(axis='x', tight=True)


    fig.tight_layout()

    return fig


