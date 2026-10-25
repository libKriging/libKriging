import pylibkriging as lk


def test_check_blas():
    # BLAS / LAPACK self-test (#429): single thread, then 2 concurrent threads
    c = lk.LinearAlgebra.check_blas(2)
    assert set(c.keys()) == {"ok", "library", "report"}
    assert c["library"]
    assert "single thread:" in c["report"]
    assert "2 concurrent threads" in c["report"]
    assert c["ok"], c["report"]


def test_check_blas_default_threads():
    c = lk.LinearAlgebra.check_blas()
    assert c["ok"], c["report"]
