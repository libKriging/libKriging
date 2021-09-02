"""pytest configuration

"""

import pytest
import sys
import contextlib

# ignore tests according to python version
collect_ignore = []
if sys.version_info[:2] < (3, 8):
    collect_ignore.append("")


@contextlib.contextmanager
def suppress(exception):
    """Suppress the desired exception"""
    try:
        yield
    except exception:
        pass


def pytest_configure():
    _test_import_pylibkriging()
    import pylibkriging

    skipif = pytest.mark.skipif
    pytest.suppress = suppress
    pytest.debug_only = skipif(pylibkriging.__build_type__ != 'Debug',
                               reason="Only for debug mode")
    pytest.direct_mapping = pytest.mark.skip # carma bug inside


def _test_import_pylibkriging():
    """Early diagnostic for test module initialization errors

    When there is an error during initialization, the first import will report the
    real error while all subsequent imports will report nonsense. This import test
    is done early (in the pytest configuration file, before any tests) in order to
    avoid the noise of having all tests fail with identical error messages.

    """
    try:
        import pylibkriging  # noqa: F401 imported but unused
    except Exception as e:
        print("Failed to import pylibkriging from pytest:")
        print("  {}: {}".format(type(e).__name__, e))
        sys.exit(1)


_test_import_pylibkriging()
