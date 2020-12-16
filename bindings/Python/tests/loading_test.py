import pylibkriging as m
import pytest


def test_version():
    assert m.__version__ == '0.1.3'


@pytest.mark.skipif(m.__build_type__ != 'Debug',
                    reason="Trivial demo only for debug mode")
def test_add():
    assert m.add(1, 2) == 3


@pytest.mark.skipif(m.__build_type__ != 'Debug',
                    reason="Trivial demo only for debug mode")
def test_substract():
    assert m.subtract(1, 2) == -1
