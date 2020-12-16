import pylibkriging as m


def test_version():
    assert m.__version__ == '0.1.3'


def test_add():
    assert m.add(1, 2) == 3


def test_substract():
    assert m.subtract(1, 2) == -1
