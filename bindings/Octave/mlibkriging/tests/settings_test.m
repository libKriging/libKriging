1;  % mark this file as a script

% docs/Settings.md (#468): set / get round trip of every user setting, with the
% values as a user types them (plain doubles), restored at the end.

settings = {
  'Optim', 'set_max_restart', 'get_max_restart', 3;
  'Optim', 'set_max_iteration', 'get_max_iteration', 57;
  'Optim', 'set_gradient_tolerance', 'get_gradient_tolerance', 0.0125;
  'Optim', 'set_objective_rel_tolerance', 'get_objective_rel_tolerance', 0.0025;
  'Optim', 'set_theta_lower_factor', 'get_theta_lower_factor', 0.25;
  'Optim', 'set_theta_upper_factor', 'get_theta_upper_factor', 4.5;
  'Optim', 'use_variogram_bounds_heuristic', 'variogram_bounds_heuristic_used', false;
  'Optim', 'use_reparametrize', 'is_reparametrized', false;
  'Optim', 'set_log_level', 'get_log_level', 2;
  'Parallel', 'set_num_threads_percent', 'get_num_threads_percent', 0.5;
  'Parallel', 'set_thread_pool_size', 'get_thread_pool_size', 2;
  'Parallel', 'set_thread_start_delay_ms', 'get_thread_start_delay_ms', 7;
  'Parallel', 'set_blas_threads_min_n', 'get_blas_threads_min_n', 250;
  'Parallel', 'set_omp_min_rows', 'get_omp_min_rows', 300;
  'LinearAlgebra', 'set_chol_jitter', 'get_chol_jitter', 1e-7;
};

for i = 1:size(settings, 1)
  cls = settings{i, 1};
  setter = str2func([cls '.' settings{i, 2}]);
  getter = str2func([cls '.' settings{i, 3}]);
  value = settings{i, 4};
  orig = getter();
  setter(value);
  got = getter();
  assert(double(got) == double(value), sprintf('%s.%s: %g ~= %g', cls, settings{i, 2}, double(got), double(value)));
  setter(orig);
  assert(double(getter()) == double(orig), sprintf('%s.%s: not restored', cls, settings{i, 2}));
end

% set_max_restart rejects a negative count and keeps the value
orig = Optim.get_max_restart();
failed = false;
try
  Optim.set_max_restart(-1);
catch
  failed = true;
end
assert(failed);
assert(Optim.get_max_restart() == orig);
