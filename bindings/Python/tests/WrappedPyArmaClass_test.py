import numpy as np
import pylibkriging as lk
import pytest


@pytest.debug_only
@pytest.mark.parametrize("n", [1, 2, 4, 5, 8, 16, 17, 32, 64, 128, 256])
def test_WrappedPyArmaClass_should_be_valid_for_any_size_in_default_order(n):
    a = np.random.random((n, 5))
    lk.WrappedPyArmaClass(a)


@pytest.debug_only
@pytest.mark.parametrize("n", [1, 2, 4, 5, 8, 16, 17, 32, 64, 128, 256])
def test_WrappedPyArmaClass_should_be_valid_for_any_size_in_f_order(n):
    a = np.reshape(np.random.random((n, 5)), (5, n), order='F')
    lk.WrappedPyArmaClass(a)


@pytest.debug_only
@pytest.mark.parametrize("n", [1, 2, 4, 5, 8, 16, 17, 32, 64, 128, 256])
def test_WrappedPyArmaClass_should_be_valid_for_any_size_in_c_order(n):
    a = np.reshape(np.random.random((n, 5)), (5, n), order='C')
    lk.WrappedPyArmaClass(a)
