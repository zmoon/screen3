# screen3
*Thin Python wrapper for the* [SCREEN3](https://www.epa.gov/scram/air-quality-dispersion-modeling-screening-models#screen3) *point source dispersion model*

[![CI workflow status](https://github.com/zmoon/screen3/actions/workflows/ci.yml/badge.svg)](https://github.com/zmoon/screen3/actions/workflows/ci.yml)
[![Version on PyPI](https://img.shields.io/pypi/v/screen3.svg)](https://pypi.org/project/screen3/)

[pdoc](https://pdoc.dev/) API documentation: <https://zmoon.github.io/screen3/screen3.html>

## Installation

Stable:
```
pip install screen3
```

Latest:
```
pip install git+https://github.com/zmoon/screen3
```

### SCREEN3

The `screen3` Python installation via `pip` does not include the SCREEN3 model itself.
To download it (into `./src/`), you can use (within Python):
```python
import screen3
screen3.download()
```
or on the command line:
```
python -c "import screen3; screen3.download()"
```

#### Windows

On Windows, the above (download only) is sufficient to be able to use `screen3`,
since EPA provides an executable compiled for Windows.

#### Non-Windows

On non-Windows platforms, the SCREEN3 model must be built from source.
The `screen3.build` function, which requires `patch` and `gfortran`, can be used:
```
python -c "import screen3; screen3.build()"
```
or combined with the download:
```
python -c "import screen3; screen3.download(); screen3.build()"
```

## Examples

To obtain the examples, `git clone` the repo, use Code > Download Zip (buttons), or use this [DownGit](https://github.com/MinhasKamal/DownGit) link:  
<https://downgit.github.io/#/home?url=https://github.com/zmoon/screen3/tree/main/examples>
