% Self-test of the detected linear algebra backend (#429): single thread, then 2 concurrent threads
[ok, backend, report] = linearalgebra_check(2);
disp(report)
assert(ischar(backend) && ~isempty(backend))
assert(strncmp(report, "backend: ", 9))
assert(~isempty(strfind(report, "single thread:")))
assert(~isempty(strfind(report, "2 concurrent threads")))
assert(ok)

% default thread count, ok only
assert(linearalgebra_check())
