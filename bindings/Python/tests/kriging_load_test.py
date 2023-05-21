import pylibkriging as lk
import pytest
from find_dir import find_reference_dir


def test_load():
    refpath = find_reference_dir()
    h5file = refpath / "kriging_example.h5"
    kr = lk.anykriging_load(h5file)
    assert isinstance(kr, lk.Kriging)
