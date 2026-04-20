import platform
import sys

# Fix "cannot allocate memory in static TLS block" error on Linux/Unix
# This occurs because OpenMP (libgomp/libomp) uses static TLS which has limited space
# when loaded via dlopen(). We preload it using RTLD_GLOBAL before importing the module.
if platform.system() != "Windows":
    import ctypes
    import os

    # Try to preload OpenMP library (libgomp or libomp) with RTLD_GLOBAL
    # This ensures its TLS is allocated in the initial TLS block
    openmp_libs = ['libgomp.so.1', 'libgomp.so', 'libomp.so']
    for lib_name in openmp_libs:
        try:
            # RTLD_GLOBAL (0x00100 | 0x00002) makes symbols available to subsequently loaded libraries
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
            break  # Successfully loaded
        except OSError:
            continue  # Try next library name
else:  # Windows
    import os

    shared_lib_paths = [os.path.join(os.path.dirname(__file__), 'shared_libs')]  # cf setup.py
    lk_path = os.environ.get("LIBKRIGING_DLL_PATH")
    if lk_path:
        for path in lk_path.split(os.pathsep):
            shared_lib_paths.append(path)

    # alternative method if lib/site-packages prefix is not reliable (requires update of setup.py)
    # import distutils  # https://docs.python.org/3/distutils/apiref.html#module-distutils.sysconfig
    # shared_lib_path = os.path.join(distutils.sysconfig.PREFIX, 'pylibkriging', 'shared_libs')

    if sys.version_info[:2] < (3, 8):  # < 3.8.0
        for path in shared_lib_paths:
            if os.path.isdir(path):
                os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
    else:
        for path in shared_lib_paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)

from _pylibkriging import *
from _pylibkriging import __version__, __build_type__

# Type alias to switch to the right binding
Kriging = WrappedPyKriging
NuggetKriging = WrappedPyNuggetKriging
NoiseKriging = WrappedPyNoiseKriging
MLPKriging = WrappedPyMLPKriging
LinearRegression = PyLinearRegression

import re as _re
import numpy as _np


def _encode_string_columns(X, warping):
    """Detect string columns in X, encode to integers, rewrite warping specs.

    Parameters
    ----------
    X : array-like (numpy array, pandas DataFrame, or list-of-lists)
        Input matrix, possibly containing string columns.
    warping : list of str
        Warp specifications, one per column.

    Returns
    -------
    X_num : numpy.ndarray of float64
        Encoded numeric matrix.
    warping_out : list of str
        Warping specs with level names injected for string columns.
    """
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            cols = [X.iloc[:, j] for j in range(X.shape[1])]
        else:
            X = _np.asarray(X)
            col_names = None
            cols = [X[:, j] for j in range(X.shape[1])]
    except ImportError:
        X = _np.asarray(X)
        col_names = None
        cols = [X[:, j] for j in range(X.shape[1])]

    warping_out = list(warping)
    n = len(cols[0])
    X_num = _np.empty((n, len(cols)), dtype=_np.float64)

    def _is_string_column(arr):
        """Check if an array column contains non-numeric string values."""
        if arr.dtype.kind in ('U', 'S'):
            # Pure string array — but could be stringified floats from column_stack
            try:
                _np.array(arr, dtype=_np.float64)
                return False  # Convertible to float, not a true string column
            except (ValueError, TypeError):
                return True
        if arr.dtype.kind == 'O':
            for v in arr[:min(10, len(arr))]:
                if isinstance(v, str):
                    try:
                        float(v)
                    except (ValueError, TypeError):
                        return True  # Truly non-numeric string
            return False
        return False

    for j, col in enumerate(cols):
        arr = _np.asarray(col)
        if _is_string_column(arr):
            # String column: build sorted label list, encode
            str_vals = [str(v) for v in arr]
            labels = sorted(set(str_vals))
            label_map = {lab: i for i, lab in enumerate(labels)}
            X_num[:, j] = _np.array([label_map[v] for v in str_vals], dtype=_np.float64)

            # Rewrite warping spec to include level names
            spec = warping_out[j]
            spec_lower = spec.strip().lower()
            # Parse: "categorical", "categorical(3)", "categorical(3,2)" -> inject names
            m = _re.match(r'^(categorical|ordinal)\s*(?:\(([^)]*)\))?$', spec_lower)
            if m:
                wtype = m.group(1)
                args = m.group(2)
                names_str = '[' + ','.join('"' + lab + '"' for lab in labels) + ']'
                if wtype == 'categorical':
                    embed_dim = 2  # default
                    if args:
                        parts = [p.strip() for p in args.split(',')]
                        if len(parts) >= 2:
                            embed_dim = int(parts[-1])
                    warping_out[j] = f'categorical({names_str},{embed_dim})'
                else:  # ordinal
                    warping_out[j] = f'ordinal({names_str})'
            else:
                raise ValueError(
                    f"Column {j} contains strings but warping spec '{spec}' "
                    f"is not 'categorical' or 'ordinal'")
        else:
            X_num[:, j] = _np.asarray(arr, dtype=_np.float64)

    return X_num, warping_out


def _has_string_columns(X):
    """Check if X contains any non-numeric string columns."""
    def _col_has_nonnumeric_strings(arr):
        for v in arr[:min(10, len(arr))]:
            if isinstance(v, str):
                try:
                    float(v)
                except (ValueError, TypeError):
                    return True
        return False

    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            for j in range(X.shape[1]):
                if X.dtypes.iloc[j].kind in ('U', 'S', 'O'):
                    if _col_has_nonnumeric_strings(X.iloc[:, j].values):
                        return True
            return False
    except ImportError:
        pass
    arr = _np.asarray(X)
    if arr.ndim == 2:
        if arr.dtype.kind in ('U', 'S'):
            # Could be stringified floats from column_stack
            for j in range(arr.shape[1]):
                try:
                    _np.array(arr[:, j], dtype=_np.float64)
                except (ValueError, TypeError):
                    return True
            return False
        if arr.dtype.kind == 'O':
            for j in range(arr.shape[1]):
                if _col_has_nonnumeric_strings(arr[:, j]):
                    return True
    return False


class WarpKriging:
    """WarpKriging with automatic string-column encoding.

    When X contains string columns (numpy object array or pandas DataFrame
    with string dtype), the corresponding columns are automatically encoded
    as integers and the warping spec is rewritten to include level names.

    All other arguments and methods are forwarded to the C++ binding.
    """

    def __init__(self, *args, warping=None, **kwargs):
        # Dispatch: (warping, kernel) or (y, X, warping, ...)
        if warping is not None and len(args) >= 2:
            # (y, X, warping=..., ...)
            y, X = args[0], args[1]
            rest = args[2:]
            if _has_string_columns(X):
                X, warping = _encode_string_columns(X, warping)
            else:
                X = _np.asarray(X, dtype=_np.float64)
            self._impl = WrappedPyWarpKriging(
                _np.asarray(y, dtype=_np.float64), X, warping, *rest, **kwargs)
        elif warping is not None:
            # (warping, kernel=...) — no data
            self._impl = WrappedPyWarpKriging(warping, **kwargs)
        elif len(args) >= 3:
            # Positional: (y, X, warping, ...)
            y, X, warping = args[0], args[1], args[2]
            rest = args[3:]
            if _has_string_columns(X):
                X, warping = _encode_string_columns(X, warping)
            else:
                X = _np.asarray(X, dtype=_np.float64)
            self._impl = WrappedPyWarpKriging(
                _np.asarray(y, dtype=_np.float64), X, warping, *rest, **kwargs)
        else:
            self._impl = WrappedPyWarpKriging(*args, **kwargs)

    def _encode_X(self, X):
        """Encode string columns in X using level names from warping specs."""
        if not _has_string_columns(X):
            return _np.asarray(X, dtype=_np.float64)
        warping = self._impl.warping()
        X_enc, _ = _encode_string_columns(X, warping)
        return X_enc

    def fit(self, y, X, regmodel="constant", normalize=False,
            optim="BFGS+Adam", objective="LL", parameters=None):
        if parameters is None:
            parameters = {}
        if _has_string_columns(X):
            warping = self._impl.warping()
            X, _ = _encode_string_columns(X, warping)
        else:
            X = _np.asarray(X, dtype=_np.float64)
        self._impl.fit(_np.asarray(y, dtype=_np.float64), X,
                       regmodel, normalize, optim, objective, parameters)

    def predict(self, X, return_stdev=True, return_cov=False, return_deriv=False):
        return self._impl.predict(self._encode_X(X),
                                  return_stdev, return_cov, return_deriv)

    def simulate(self, nsim=1, seed=123, X=None, will_update=False):
        if X is not None:
            X = self._encode_X(X)
        return self._impl.simulate(nsim, seed, X, will_update)

    def update(self, y_u, X_u, refit=True):
        self._impl.update(
            _np.asarray(y_u, dtype=_np.float64),
            self._encode_X(X_u), refit)

    def update_simulate(self, y_u, X_u):
        return self._impl.update_simulate(
            _np.asarray(y_u, dtype=_np.float64),
            self._encode_X(X_u))

    def copy(self):
        c = WarpKriging.__new__(WarpKriging)
        c._impl = self._impl.copy()
        return c

    def save(self, filename):
        self._impl.save(filename)

    @staticmethod
    def load(filename):
        obj = WarpKriging.__new__(WarpKriging)
        obj._impl = WrappedPyWarpKriging.load(filename)
        return obj

    # Forward all other methods directly
    def summary(self): return self._impl.summary()
    def logLikelihood(self): return self._impl.logLikelihood()
    def logLikelihoodFun(self, theta, **kw): return self._impl.logLikelihoodFun(theta, **kw)
    def kernel(self): return self._impl.kernel()
    def X(self): return self._impl.X()
    def centerX(self): return self._impl.centerX()
    def scaleX(self): return self._impl.scaleX()
    def y(self): return self._impl.y()
    def centerY(self): return self._impl.centerY()
    def scaleY(self): return self._impl.scaleY()
    def normalize(self): return self._impl.normalize()
    def regmodel(self): return self._impl.regmodel()
    def F(self): return self._impl.F()
    def T(self): return self._impl.T()
    def M(self): return self._impl.M()
    def z(self): return self._impl.z()
    def beta(self): return self._impl.beta()
    def theta(self): return self._impl.theta()
    def sigma2(self): return self._impl.sigma2()
    def is_fitted(self): return self._impl.is_fitted()
    def feature_dim(self): return self._impl.feature_dim()
    def warping(self): return self._impl.warping()
