# screen3
*Thin Python wrapper for the* [SCREEN3](https://www.epa.gov/scram/air-quality-dispersion-modeling-screening-models#screen3) *point source dispersion model*

[![CI workflow status](https://github.com/zmoon/screen3/actions/workflows/ci.yml/badge.svg)](https://github.com/zmoon/screen3/actions/workflows/ci.yml)

[pdoc](https://pdoc.dev/) API documentation: <https://zmoon.github.io/screen3/screen3.html>

## Installation

```
pip install git+https://github.com/zmoon/screen3
```

This does not include the SCREEN3 model itself.
To download it, you can use (within Python):
```python
import screen3
screen3.download()
```

On Windows, the above is sufficient to be able to use `screen3`.
On non-Windows platforms, the model must be built from source.
The [`src/build.sh`](src/build.sh) script can be used.
