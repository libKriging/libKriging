import numpy as np
import pylibkriging as lk
import pytest


@pytest.debug_only
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_direct_binding_should_be_valid_for_any_size(n):
    a = np.arange(0, n, 1)
    b = np.arange(1, n + 1, 1)

    result = lk.direct_binding(a, b)
    ref = a + b
    assert np.array_equal(result, ref)
