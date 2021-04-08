import screen3


def test_examples_run():
    import importlib
    import sys
    from pathlib import Path

    examples = Path(__file__).parent / "examples"
    assert examples.is_dir()
    sys.path.append(str(examples))

    for p in examples.glob("*.py"):
        importlib.import_module(p.stem)
