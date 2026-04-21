startup;
cfg.n = 10; cfg.p = 0.75; cfg.q = 0.2; cfg.sigma = 0.01; cfg.seed = 7; cfg.fast = true; cfg.run_fused_ta = true; cfg.alignment = 'robust'; cfg.model = 'uniform';
S = run_methods_synthetic(cfg);
assert(height(S) == 5);
assert(all(~isnan(S.median_error)));
params = default_cycle_params();
assert(params.tmax == 20 && params.beta == 20 && abs(params.delta - 1e-8) < eps);
fprintf('MATLAB smoke test completed.\n');
