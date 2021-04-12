# -*- coding: utf-8 -*-
"""
Python functions for using
[SCREEN3](https://www.epa.gov/scram/air-quality-dispersion-modeling-screening-models#screen3).
* `screen3.run` – wrapper to quickly create a run without needing to go through the prompt
* `screen3.read` – load the output into a pandas DataFrame
* `screen3.plot` – some plotting routines using the output DataFrame

Most of the variables related to SCREEN3 are the same as the ones in the SCREEN3 input and output files,
including that they are in uppercase.

`screen3` lives [on GitHub](https://github.com/zmoon/screen3).

.. note::
   Python 3.6+ is required
"""
# TODO: type annotations for the main user-facing fns?

import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

import numpy as np

__all__ = (
    "run",
    "read",
    "plot",
    "download",
    "build",
    "load_example",
    "SCREEN_OUT_COL_UNITS_DICT",
    "DEFAULT_SRC_DIR",
)


# Check for Python 3.6+
# Note: `sys.version_info` became a named tuple in 3.1
if not (sys.version_info.major >= 3 and sys.version_info.minor >= 6):
    raise Exception(f"Python version must be 3.6+.\nYours is {sys.version}")


_THIS_DIR = Path(__file__).parent


# DEFAULT_SRC_DIR = Path.home() / ".local/screen3/src"
DEFAULT_SRC_DIR = "./src"
"""Default directory in which to place the SCREEN3 source code from EPA."""

_DEFAULT_EXE_PATH = f"{DEFAULT_SRC_DIR}/SCREEN3.exe"


def download(*, src=DEFAULT_SRC_DIR):
    """Download the SCREEN3 zip from EPA and extract to directory `src`.

    If it fails, you can always download it yourself some other way.

    <https://gaftp.epa.gov/Air/aqmg/SCRAM/models/screening/screen3/screen3.zip>

    Parameters
    ----------
    src : path-like
        Where to extract the files to.
        .. note::
           As long as this isn't modified, the `src`/`exe` keyword doesn't need to be changed
           for `build`, `run`, or `load_example`.
    """
    import io
    import zipfile
    from pathlib import Path

    import requests

    url = "https://gaftp.epa.gov/Air/aqmg/SCRAM/models/screening/screen3/screen3.zip"

    extract_to = Path(src)
    extract_to.mkdir(exist_ok=True, parents=True)
    
    r = requests.get(url, verify=False)  # TODO: get it working without having to disable certificate verification
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for info in z.infolist():
            with z.open(info) as zf, open(extract_to / info.filename, "wb") as f:
                f.write(zf.read())


_SCREEN3A_PATCH = """
386,397c386,401
< $IF DEFINED (LAHEY)
<       CALL DATE(RUNDAT)
<       CALL TIME(RUNTIM)
< $ELSE
<       CALL GETDAT(IPTYR, IPTMON, IPTDAY)
<       CALL GETTIM(IPTHR, IPTMIN, IPTSEC, IPTHUN)
< C     Convert Year to Two Digits
<       IPTYR = IPTYR - 100*INT(IPTYR/100)
< C     Write Date and Time to Character Variables, RUNDAT & RUNTIM
<       WRITE(RUNDAT,'(2(I2.2,1H/),I2.2)') IPTMON, IPTDAY, IPTYR
<       WRITE(RUNTIM,'(2(I2.2,1H:),I2.2)') IPTHR, IPTMIN, IPTSEC
< $ENDIF
---
> !$IF DEFINED (LAHEY)
> !      CALL DATE(RUNDAT)
> !      CALL TIME(RUNTIM)
> !$ELSE
> !      CALL GETDAT(IPTYR, IPTMON, IPTDAY)
> !      CALL GETTIM(IPTHR, IPTMIN, IPTSEC, IPTHUN)
> !C     Convert Year to Two Digits
> !      IPTYR = IPTYR - 100*INT(IPTYR/100)
> !C     Write Date and Time to Character Variables, RUNDAT & RUNTIM
> !      WRITE(RUNDAT,'(2(I2.2,1H/),I2.2)') IPTMON, IPTDAY, IPTYR
> !      WRITE(RUNTIM,'(2(I2.2,1H:),I2.2)') IPTHR, IPTMIN, IPTSEC
> !$ENDIF
> 
> ! here for GNU (zm)
>       call DATE_AND_TIME(DATE=RUNDAT, TIME=RUNTIM)
> 
""".lstrip()


_DEPVAR_PATCH = """
35c35
< 
\ No newline at end of file
---
> 
""".lstrip()


def build(*, src=DEFAULT_SRC_DIR):
    """Build the SCREEN3 executable by pre-processing the sources and compiling with GNU Fortran.
    
    .. note::
       Requires `dos2unix` (for Linux/macOS), `patch`, and `gfortran` on PATH.

    Parameters
    ----------
    src : path-like
        Source directory, containing `SCREEN3A.FOR` etc., e.g., downloaded using `screen3.download`.
    """
    cwd = Path.cwd()
    bld = Path(src)

    os.chdir(bld)

    srcs = ['SCREEN3A.FOR', 'SCREEN3B.FOR', 'SCREEN3C.FOR']

    # Fix line endings
    if platform.system() != "Windows":
        subprocess.run(['dos2unix'] + srcs)

    # Patch code
    with open("SCREEN3A.FOR.patch", "w") as f:
        f.write(_SCREEN3A_PATCH)
    with open("DEPVAR.INC.patch", "w") as f:
        f.write(_DEPVAR_PATCH)
    subprocess.run(['patch', 'SCREEN3A.FOR', 'SCREEN3A.FOR.patch'])
    subprocess.run(['patch', 'DEPVAR.INC', 'DEPVAR.INC.patch'])

    # Compile
    subprocess.run(['gfortran', '-cpp'] + srcs + ['-o', 'SCREEN3.exe'])

    os.chdir(cwd)


# note that U10M becomes UHANE in non-regulatory mode
SCREEN_OUT_COL_NAMES = 'DIST CONC STAB U10M USTK MIX_HT PLUME_HT SIGMA_Y SIGMA_Z DWASH'.split()
SCREEN_OUT_COL_UNITS = ['m', 'μg/m$^3$', '', 'm/s', 'm/s', 'm', 'm', 'm', 'm', '']
SCREEN_OUT_COL_UNITS_DICT = dict(zip(SCREEN_OUT_COL_NAMES, SCREEN_OUT_COL_UNITS))
"""Dict of units for the outputs to be used in the plots. For example, `'DIST': 'm'`."""


def read(
    fp, 
    *, 
    t_run=None,
    run_inputs=None,
):
    """Read and extract data from a SCREEN3 run (i.e., a `SCREEN.OUT` file).
    
    Parameters
    ----------
    fp : str, pathlib.Path
        File path to the `SCREEN.OUT` file,
        e.g., `'./screen3/SCREEN.OUT'`.
    t_run : datetime.datetime
        Time of the run.
        If the output file (`SCREEN.OUT`) is older than this, the run did not complete successfully.
    run_inputs : dict, optional
        `run` passes this so that we can store what the inputs to the Python function were for this run,
        though `SCREEN.OUT` should have much of the same info.

    Returns
    -------
    df : pd.DataFrame
        SCREEN3 output dataset.

        If the pandas version has [`attrs`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.attrs.html),
        `df.attrs` will include the following:
        * `'units'`: (dict) *keys*: the variables names, *values*: unit strings (for use in plots)
        * `'SCREEN_OUT'`: (str) the SCREEN3 output file as a string
        * `'SCREEN_DAT'`: (str) the SCREEN3 DAT (copied inputs) file as a string
        * `'run_inputs'`: (dict) copy of the inputs to the the `run` function used to generated this dataset (if applicable)
    """
    import datetime
    
    import pandas as pd

    if run_inputs is None:
        run_inputs = {}

    # If we have `t_run`, first check that the file has indeed been modified
    p = Path(fp)
    t_out_modified = datetime.datetime.fromtimestamp(p.stat().st_mtime)
    if t_run is not None and p.is_file():  # i.e., t_run has been specified
        if (t_run - t_out_modified) > datetime.timedelta(seconds=1):  
            raise ValueError(
                "`SCREEN.OUT` is older than time of the run."
                " This probably means that SCREEN failed to run because it didn't like the value of one or more inputs."
                " Or some other reason.")

    # Read lines
    with open(fp, 'r') as f:
        lines_raw = f.readlines()
    lines_stripped = [line.strip() for line in lines_raw]

    # TODO: could also read other info from the file like the run settings at the top

    iheader = [i for i, s in enumerate(lines_stripped) if s.split()[:2] == ['DIST', 'CONC']]
    if not iheader:
        raise ValueError(
            "Data table not found in output. "
            "This probably means that SCREEN failed to complete the run because "
            "it didn't like the value of one or more inputs. "
            "Or some other reason."
        )

    if len(iheader) > 1:
        print(f"Multiple sets of data found. Only loading the first one.")
    iheader = iheader[0]

    # Find first blank after iheader
    iblankrel = lines_stripped[iheader:].index("")
    iblank = iheader + iblankrel

    # Could (technically) read these from file instead
    col_names = SCREEN_OUT_COL_NAMES
    col_units = SCREEN_OUT_COL_UNITS

    # Load into pandas
    # Note: at some point `pd.read_table` was deprecated but seems like that was undone?
    n_X = iblank - iheader - 2
    df = pd.read_table(fp, 
        sep=None, skipinitialspace=True, engine='python',
        keep_default_na=False,
        skiprows=iheader+2, nrows=n_X)
    df.columns = col_names
    
    units_dict = dict(zip(col_names, col_units))

    p_dat = p.parents[0] / f'{p.stem}.DAT'
    try:
        with open(p_dat, 'r') as f:
            s_screen_dat = f.read()
    except FileNotFoundError:
        s_screen_dat = None

    # Assign attrs to df (pandas 1.0+)
    if hasattr(df, "attrs"):
        df.attrs.update({
            'SCREEN_DAT': s_screen_dat,
            'SCREEN_OUT': ''.join(lines_raw),
            'units': units_dict,
            'run_inputs': run_inputs
        })
    
    return df


def run(
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
    exe=_DEFAULT_EXE_PATH,
):
    """Create SCREEN3 input file, feed it to the executable, and load the result.

    .. note::
       Inputs must be specified as keyword arguments, but can be entered in any order (non-positional).

    Parameters
    ----------
    Q : float 
        Emission rate (g/s).
    HS : float 
        Stack height (m).
    DS : float 
        Stack inside diameter (m).
    VS : float 
        Stack gas exit velocity (m/s).
    TS : float 
        Stack gas temperature (K).
    TA : float 
        Ambient air temperature (K).
    ZR : float 
        Receptor height above ground (flagpole receptor) (M).
    X : array_like
        Array of downwind distances at which to compute the outputs (m).
    IMETEO : int, {1, 2, 3}
        1. Full (supplying stability class + WS for each grid point), 
           or SCREEN runs for each STAB option??...
        2. Single stability class
        3. Stability class + WS
    ISTAB : int
        Stability class.
        1 (= A) to 6 (= F).
    WS : float
        Mean wind speed at 10 m (m/s).
        .. warning::
           The run will fail if you exceed the maximum that SCREEN3 allows!
    U_or_R : str {'U', 'R'}
        Urban (U) or rural (R).
    DOWNWASH_YN : str {'Y', 'N'}
        whether to apply building downwash calculations
        the building dimension parameters do nothing if it is 'N'
    HB : float
        Building height (m).
    HL : float
        Minimum horizontal building dimension (m).
    HW : float
        Maximum horizontal building dimension (m).
    exe : path-like
        Path to the executable to use, e.g., as a `str` or `pathlib.Path`.

    Examples
    --------
    Change parameters.
    ```python
    screen3.run(TA=310, WS=2)
    ```
    Specify executable to use if yours isn't in the default place (`./src/SCREEN3.exe`).
    ```python
    screen3.run(exe="/path/to/executable")
    ```

    Returns
    -------
    df : pd.DataFrame
        Results dataset, read from the `SCREEN.OUT` by `read`.

    Notes
    -----
    The SCREEN3 program parses the inputs and makes a copy called `SCREEN.DAT`.
    Upon running, it produces an output file called `SCREEN.OUT`.
    Both of these will be in the source directory, where the executable resides.
    """
    import datetime

    inputs = locals()  # collect inputs for saving in the df

    # TODO: should validate wind speed?

    # Check exe is file
    exe = Path(exe)
    if not exe.is_file():
        raise ValueError(f"{exe.absolute()!r} does not exist or is not a file.")

    # Check for H changes without downwad
    H_defaults = (30.0, 10.0, 20.0)  # keep in sync with fn defaults
    if any(x0 != x for x0, x in zip(H_defaults, [HB, HL, HW])) and DOWNWASH_YN == 'N':
        print("Note: You have modified a building parameter, but downwash is not enabled.")

    # Check x positions
    X = np.asarray(X)
    if X.size > 50:
        raise ValueError('SCREEN3 does not support inputting more than 50 distance values')

    # ------------------------------------------------------
    # Other parameters
    # For now the user cannot set these
    # although they are required inputs to the model
    
    I_SOURCE_TYPE = 'P'  # point source

    # Complex terrain also currently not used
    # (assumed flat)
    TERRAIN_1_YN = 'N'  # complex terrain screen above stack height
    TERRAIN_2_YN = 'N'  # simple terrain above base
    # HTER = 10.  # MAXIMUM TERRAIN HEIGHT ABOVE STACK BASE (M)
    # ^ not needed if not using the terrain settings
    # at least for simple terrain, seems to just run the x positions for different heights 

    # Currently user input of distances is required
    # i.e., automated option not used
    AUTOMATED_DIST_YN = 'N'
    DISCRETE_DIST_YN = 'Y'

    # Final questions
    FUMIG_YN = 'N'
    PRINT_YN = 'N'

    # ------------------------------------------------------
    # Create the input file

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
    """.strip() + "\n"
    # ^ need newline at end or Fortran complains when passing text on the cl

    # TODO: optionally write the input text file to disk in cwd
    # ifn = 'screen3_input.txt'  # input filename
    # ifp = cwd / ifn
    # with open(ifn, 'w') as f: 
    #     f.write(dat_text)

    # ------------------------------------------------------
    # Run the SCREEN executable

    t_utc_run = datetime.datetime.now()

    s_exe = str(exe.absolute())
    cwd = Path.cwd()
    src_dir = exe.parent

    # Move to src location so that the output file will be saved there
    os.chdir(src_dir)

    # Invoke executable
    subprocess.run(
        args=[s_exe],
        input=dat_text,
        universal_newlines=True,  # equivalent to `text=True`, but that was added in 3.7
        check=True,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Return to cwd
    os.chdir(cwd)

    # Read and parse the output
    df = read(src_dir/'SCREEN.OUT', t_run=t_utc_run, run_inputs=inputs)

    return df


def _add_units(x_units, y_units, *, ax=None):
    """Add units and make room."""
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    xl = ax.get_xlabel()
    yl = ax.get_ylabel()
    sxu = f'({x_units})' if x_units else ''
    syu = f'({y_units})' if y_units else ''
    ax.set_xlabel(f'{xl} {sxu}' if xl else sxu)
    ax.set_ylabel(f'{yl} {syu}' if xl else syu)


def plot(
    df, 
    *, 
    labels=None,
    yvals=None,
    yvar="",
    yvar_units="",
    plot_type='line', 
    ax=None,
    **pyplot_kwargs):
    """Plot concentration.

    The first argument (positional) can be one `df` or a list of `df`s. 
    If you pass a list of `df`s, you can also pass `labels`. 

    The default plot type is `'line'`, but 2-D plots `'pcolor'` and `'contourf'` are also options. 
    Unless you additionally pass `yvals`, they will be plotted as if the labels were equally spaced. 

    Parameters
    ----------
    df : pd.DataFrame or list of pd.DataFrame
        Data extracted from the `SCREEN.OUT` of the run(s) using `read`.
    labels : array_like
        Labels for the separate cases.
        Used if input `df` is a list instead of just one dataset.
        Currently we assume that the first part of the label is the variable name that is being varied.
    yvals : array_like
        Optional positions of labels.
    yvar : str
        *y* variable name.
        Only used if labels not provided.
    yvar_units : str
        Units to use when labeling `yvar`.
        Only used if `labels` not provided.
    plot_type : str {'line', 'pcolor', 'contourf'}
        The type of plot to do.
    ax : plt.Axes
        If an existing ax is not passed, we create a new figure (and ax).
        Else, we plot on the existing ax.
    **pyplot_kwargs
        Passed on to the relevant pyplot plotting function,
        e.g., `ax.plot`.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    if labels is None:
        labels = []
    if yvals is None:
        yvals = []

    units = SCREEN_OUT_COL_UNITS_DICT
    # or `df.attrs['units']`

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
                _add_units(units['DIST'], yvar_units, ax=ax)

            else:
                Y = labels
                _add_units(units['DIST'], '', ax=ax)

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
        _add_units(units['DIST'], units['CONC'], ax=ax)  # use the units dict to look up the units

        # ax.set_xlim(xmin=0, xmax=)
        ax.autoscale(axis='x', tight=True)

    fig.tight_layout()

    return fig


def load_example(s, *, src=DEFAULT_SRC_DIR):
    """Load one of the examples included with the `screen3.zip` download,
    such as `'EXAMPLE.OUT'` from SCREEN3 source directory `src`.

    Examples
    --------
    ```python
    screen3.load_example("EXAMPLE.OUT")
    ```
    """
    valid_examples = [
        "EXAMPLE.OUT", "examplenew.out", "examplnrnew.out",
        "EXAMPNR.OUT", "exampnrnew.out",
    ]
    if s not in valid_examples:
        raise ValueError(f"invalid example file name. Valid options are: {valid_examples}")

    return read(Path(src) / s)
