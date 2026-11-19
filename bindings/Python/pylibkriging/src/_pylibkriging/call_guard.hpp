#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_CALL_GUARD_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_CALL_GUARD_HPP

// The model methods that compute (fit, predict, simulate, update,
// update_simulate) run the libKriging call without the GIL, so that calls on
// different models from several Python threads (ThreadPoolExecutor, joblib
// threads) run at the same time, sharing the thread budget
// (docs/dev/parallel.md, 3.1). The arguments are converted before, with the
// GIL (numpy -> Armadillo), and the results after.
//
// Calls on the SAME model object are serialized by a per-object mutex: the GIL
// used to serialize them, and libKriging does not support one object used from
// two threads at the same time. Lock order: the GIL is released first, then the
// mutex taken, so a thread waiting for the mutex never holds the GIL (the
// deprecation handler re-acquires the GIL inside the call).
//
// Allocations made without the GIL go to std::malloc (gil_aware_malloc,
// pylibkriging.cpp); numpy blocks freed without it are queued and freed at the
// next call made with the GIL.

#include <pybind11/pybind11.h>

#include <mutex>

/// Mutex of a model wrapper; copies and moves get a fresh one (a copied model
/// is another object).
struct CallMutex {
  std::mutex m;
  CallMutex() = default;
  CallMutex(const CallMutex&) {}
  CallMutex(CallMutex&&) noexcept {}
  CallMutex& operator=(const CallMutex&) { return *this; }
  CallMutex& operator=(CallMutex&&) noexcept { return *this; }
};

/// Runs f() without the GIL, holding the model's mutex. f must not touch any
/// Python object.
template <typename F>
auto without_gil(CallMutex& mutex, F&& f) -> decltype(f()) {
  pybind11::gil_scoped_release release;  // first: never wait for the mutex holding the GIL
  std::lock_guard<std::mutex> lock(mutex.m);
  return f();
}

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_CALL_GUARD_HPP
