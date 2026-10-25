function [ok, backend, report] = linearalgebra_check(n_threads)
    % linearalgebra_check - Self-test of the linear algebra backend
    %
    %   [ok, backend, report] = linearalgebra_check(n_threads)
    %
    % Calls the linear algebra routines libKriging relies on (crossprod,
    % Cholesky, triangular and general solves, inverse, rcond, QR, Woodbury
    % solve, symmetric eigen decomposition) and compares each result with an
    % oracle computed in long double without the BLAS, from one thread then
    % from n_threads concurrent threads (default 0: hardware concurrency). The
    % BLAS / LAPACK backend is detected, not chosen.
    %   ok: true when every result is within its rounding bound
    %   backend: detected backend and library libKriging's dgemm is bound to
    %   report: one line per check, then the workaround of a backend known
    %     to be faulty
    if nargin < 1
        n_threads = 0;
    end
    [ok, backend, report] = mLibKriging("LinearAlgebra::check", int32(n_threads));
end
