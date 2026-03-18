import pylibkriging as lk
import numpy as np


def test_random():
    g = lk.RandomGenerator(123)
    expected_result = np.array([[0.71295532, 0.78002776],
                                [0.42847093, 0.41092437],
                                [0.69088485, 0.5796943],
                                [0.71915031, 0.13995076],
                                [0.49111893, 0.40101755]])
    generated_result = g.uniform(5, 2)
    # print(generated_result)
    assert np.isclose(generated_result, expected_result).all()
