import platform
from pathlib import Path

import pytest

import screen3


def test_examples_run():
    import importlib
    import sys

    examples = Path(__file__).parent / "examples"
    assert examples.is_dir()
    sys.path.append(str(examples))

    for p in examples.glob("*.py"):
        importlib.import_module(p.stem)


@pytest.mark.skipif(platform.system() != "Windows", reason="On Windows, we don't have to build")
def test_run_src_dir_with_spaces():
    import shutil
    import warnings

    import urllib3

    p = Path(__file__).parent / "src dir with spaces"
    p.mkdir(exist_ok=True)

    with pytest.warns(urllib3.exceptions.InsecureRequestWarning):
        screen3.download(extract_to=p)

    exe0 = screen3._SCREEN_EXE_PATH
    screen3.set_exe_path(p / "SCREEN3.EXE")
    screen3.run()
    screen3.set_exe_path(exe0)

    shutil.rmtree(p)
