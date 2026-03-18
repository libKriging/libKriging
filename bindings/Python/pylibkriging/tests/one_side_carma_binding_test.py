import numpy as np
import pylibkriging as lk
import pytest


@pytest.debug_only
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_one_side_carma_binding_should_be_valid_for_any_size(n):
    a = np.arange(0, n, 1)
    b = np.arange(1, n + 1, 1)

    result = lk.one_side_carma_binding(a, b)
    ref = a + b
    assert np.linalg.norm(result - ref) < 1e-16, f"Bad result\nresult={result}\nref={ref}\ndiff={result - ref}"
