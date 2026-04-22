function params = default_cycle_params(varargin)
params.tmax = 20;
params.beta = 20;
params.delta = 1e-8;
params.loss_a = 4;
params.lambda_offset = 10;
params.init_weight_scale = 20;
params.taab_nsample = 200;
params.cycle_nsample = 100;
params.sinmin = 0.6;
params.taab_iters = 5;
params.wls_max_iter = 200;
params.wls_tol = 1e-8;
params.seed = 123;
params.use_taab_init = true;
params.normalize_taab_by_pi = true;
params.no_cycle_value = 1;
params.score_clip_min = 1e-8;
params.score_clip_max = 10;
if nargin > 0
    opts = varargin{1};
    names = fieldnames(opts);
    for k = 1:numel(names)
        params.(names{k}) = opts.(names{k});
    end
end
end
