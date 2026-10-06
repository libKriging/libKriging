import os
import subprocess
import sys

import pylibkriging as lk


def test_gpu_api_is_always_defined():
    assert isinstance(lk.gpu_compiled_backends(), str)
    assert isinstance(lk.gpu_available(), bool)
    assert lk.gpu_backend() in ("none", "cuda", "hip", "sycl", "metal")
    assert lk.gpu_enabled() == (lk.gpu_backend() != "none")


def test_set_gpu_enabled_round_trip():
    initial = lk.gpu_enabled()
    try:
        lk.set_gpu_enabled(False)
        assert not lk.gpu_enabled()
        assert lk.gpu_backend() == "none"
        lk.set_gpu_enabled(True)
        # never turns on a backend without a device
        assert lk.gpu_enabled() == lk.gpu_available()
    finally:
        lk.set_gpu_enabled(initial)


def test_env_var_disables_gpu_by_default():
    code = "import pylibkriging as lk; print(lk.gpu_enabled(), lk.gpu_backend())"
    env = dict(os.environ, LK_ITERATIVE_GPU="0")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split()[-2:] == ["False", "none"]
