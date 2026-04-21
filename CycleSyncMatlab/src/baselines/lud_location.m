function result = lud_location(AdjMat,edges,gamma,opts)
if nargin < 4, opts = struct(); end
if ~isfield(opts,'maxit'), opts.maxit = 20; end
if ~isfield(opts,'delt'), opts.delt = 1e-16; end
if ~isfield(opts,'tol'), opts.tol = 1e-5; end
n = size(AdjMat,1); m = size(edges,1);
tic;
weights = ones(1,m);
old = 1; stag = 0; costs = [];
sol = [];
opts_wls.alpha_lower = 1; opts_wls.max_iter = 200; opts_wls.tol = 1e-8;
for it = 1:opts.maxit
    sol = solve_translation_wls(edges,gamma,weights,n,opts_wls);
    r = sol.residual_norms;
    weights = 1./sqrt(r.^2 + opts.delt);
    cost = sum(r); costs(end+1) = cost; %#ok<AGROW>
    if abs(old-cost)/max(abs(old),1e-15) <= opts.tol
        stag = stag + 1;
    else
        stag = 0;
    end
    old = cost;
    if stag > 5, break; end
end
result.t = sol.t;
result.runtime = toc;
result.info.iterations = numel(costs);
result.info.costs = costs;
end
