% BLAS / LAPACK self-test (#429): single thread, then 2 concurrent threads
[ok, library, report] = LinearAlgebra.check_blas(2);
disp(report)
assert(ischar(library) && ~isempty(library))
assert(~isempty(strfind(report, "single thread:")))
assert(~isempty(strfind(report, "2 concurrent threads:")))
assert(ok)

% default thread count, ok only
assert(LinearAlgebra.check_blas())
