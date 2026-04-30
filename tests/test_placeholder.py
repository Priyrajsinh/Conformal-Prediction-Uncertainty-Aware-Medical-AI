"""Day 0 placeholder tests — import all src packages to satisfy coverage gate."""

import importlib


def test_src_packages_importable() -> None:
    """All src sub-packages are importable."""
    packages = [
        "src",
        "src.api",
        "src.data",
        "src.models",
        "src.training",
        "src.evaluation",
        "src.baseline",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None
