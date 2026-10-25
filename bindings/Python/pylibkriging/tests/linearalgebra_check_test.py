import pylibkriging as lk


def test_linearalgebra_check():
    # self-test of the detected BLAS / LAPACK backend (#429): single thread, then 2 concurrent threads
    c = lk.linearalgebra_check(2)
    assert set(c.keys()) == {"ok", "backend", "report"}
    assert c["backend"]
    assert c["report"].startswith("backend: ")
    assert "single thread:" in c["report"]
    assert "2 concurrent threads" in c["report"]
    assert c["ok"], c["report"]


def test_linearalgebra_check_default_threads():
    c = lk.linearalgebra_check()
    assert c["ok"], c["report"]
