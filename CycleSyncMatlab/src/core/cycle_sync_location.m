function result = cycle_sync_location(AdjMat,edges,gamma,params)
if nargin < 4 || isempty(params), params = default_cycle_params(); end
tic;
n = size(AdjMat,1); m = size(edges,1);
taab = truncated_aab_scores(AdjMat,edges,gamma,params);
if params.use_taab_init
    weights = exp(-params.init_weight_scale*taab.scores);
else
    weights = ones(1,m);
end
alpha = ones(1,m);
residuals = ones(1,m);
sc = taab.scores;
t = zeros(3,n);
history = struct([]);
opts_wls.alpha_lower = 1;
opts_wls.max_iter = params.wls_max_iter;
opts_wls.tol = params.wls_tol;
for it = 1:params.tmax
    sol = solve_translation_wls(edges,gamma,weights,n,opts_wls);
    t = sol.t; alpha = sol.alpha; residuals = sol.residual_norms;
    sc = cycle_scores_from_locations(AdjMat,edges,gamma,t,residuals,params,it);
    lam = it/(it + params.lambda_offset);
    h = (1-lam)*residuals + lam*sc;
    h = min(max(h, params.score_clip_min), params.score_clip_max);
    weights = exp(-params.loss_a*h)./(h + params.delta);
    weights(~isfinite(weights)) = 0;
    if max(weights) <= 0, weights = ones(1,m); end
    history(it).iter = it;
    history(it).lambda = lam;
    history(it).median_residual = median(residuals);
    history(it).median_cycle_score = median(sc);
    history(it).min_weight = min(weights);
    history(it).max_weight = max(weights);
    history(it).wls_status = sol.status;
    history(it).wls_cost = sol.cost;
end
result.t = t;
result.alpha = alpha;
result.weights = weights;
result.residual_norms = residuals;
result.cycle_scores = sc;
result.init_scores = taab.scores;
result.history = history;
result.runtime = toc;
result.params = params;
end
