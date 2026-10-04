% Gpu: backend-agnostic switch for the GPU-accelerated iterative path.
assert(ischar(Gpu.compiled_backends()));
assert(islogical(Gpu.available()));
assert(any(strcmp(Gpu.backend(), {'none', 'cuda', 'hip', 'sycl', 'metal'})));
assert(Gpu.enabled() == ~strcmp(Gpu.backend(), 'none'));

initial = Gpu.enabled();
Gpu.set_enabled(false);
assert(~Gpu.enabled());
assert(strcmp(Gpu.backend(), 'none'));
Gpu.set_enabled(true);
assert(Gpu.enabled() == Gpu.available());  % never turns on a missing device
Gpu.set_enabled(initial);
