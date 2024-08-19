import numpy as np
import pylibkriging as lk
import pytest

d = {
    'bool': True,
    'int': 3,
    'float': 3.14,
    'float64': np.float64(3.1415),
    'str': 'string',
    'badcolvec': np.array([[1, 2, 3, 4]]),
    'colvec1D': np.array([1, 2, 3, 4]),
    'colvec2D': np.array([[1, 2, 3, 4]]).T,
    'rowvec': np.array([[1, 2, 3, 4]]),
    'mat': np.array([[1, 2, 3], [4, 5, 6]]),
}


@pytest.debug_only
def test_check_bool_in_dict():
    assert lk.check_dict_entry(d, "bool", "bool")


@pytest.debug_only
def test_check_int_in_dict():
    assert lk.check_dict_entry(d, "int", "int")


@pytest.debug_only
def test_check_float_in_dict():
    assert lk.check_dict_entry(d, "float", "float")

@pytest.debug_only
def test_check_float_in_dict():
    assert lk.check_dict_entry(d, "float64", "float64")


@pytest.debug_only
def test_check_str_in_dict():
    assert lk.check_dict_entry(d, "str", "str")


@pytest.debug_only
def test_check_bad_colvec_in_dict():
    with pytest.raises(ValueError):
        lk.check_dict_entry(d, "badcolvec", "colvec")


@pytest.debug_only
def test_check_colvec_in_dict():
    assert lk.check_dict_entry(d, "colvec1D", "colvec")


@pytest.debug_only
def test_check_colvec_in_dict():
    assert lk.check_dict_entry(d, "colvec2D", "colvec")


@pytest.debug_only
def test_check_mat_in_dict():
    assert lk.check_dict_entry(d, "mat", "mat")


@pytest.debug_only
def test_check_rowvec_in_dict():
    assert lk.check_dict_entry(d, "rowvec", "rowvec")


@pytest.debug_only
def test_check_undefined_key_in_dict():
    assert not lk.check_dict_entry(d, "undef", "str")


@pytest.debug_only
def test_check_existing_key_with_wrong_type_in_dict():
    with pytest.raises(ValueError):
        lk.check_dict_entry(d, "bool", "int")


@pytest.debug_only
def test_check_cast_int_to_float_in_dict():
    lk.check_dict_entry(d, "int", "float")


@pytest.debug_only
def test_check_do_not_cast_float_to_int_in_dict():
    with pytest.raises(ValueError):
        lk.check_dict_entry(d, "float", "int")
