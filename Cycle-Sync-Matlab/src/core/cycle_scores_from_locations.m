function scores = cycle_scores_from_locations(AdjMat,edges,gamma,t,residuals,params,iter)
if nargin < 7, iter = 1; end
if isfield(params,'seed') && ~isempty(params.seed), rng(params.seed + 7919*iter); end
n = size(AdjMat,1); m = size(edges,1);
nsample = params.cycle_nsample;
L = edge_lookup(edges,n);
scores = params.no_cycle_value*ones(1,m);
for e = 1:m
    i = edges(e,1); j = edges(e,2);
    c = find(AdjMat(:,i) > 0 & AdjMat(:,j) > 0);
    c = c(c ~= i & c ~= j);
    if isempty(c), continue; end
    picks = c(randi(numel(c), nsample, 1));
    vals = zeros(nsample,1);
    w = zeros(nsample,1);
    for s = 1:nsample
        k = picks(s);
        eik = abs(L(i,k)); ejk = abs(L(j,k));
        gij = gamma(:,e);
        gjk = oriented_gamma(gamma,L,j,k);
        gki = oriented_gamma(gamma,L,k,i);
        lij = norm(t(:,i) - t(:,j));
        ljk = norm(t(:,j) - t(:,k));
        lki = norm(t(:,k) - t(:,i));
        vals(s) = norm(lij*gij + ljk*gjk + lki*gki);
        w(s) = exp(-params.beta*(residuals(eik) + residuals(ejk)));
    end
    sw = sum(w);
    if sw > 1e-300 && isfinite(sw)
        scores(e) = sum(w.*vals)/sw;
    else
        scores(e) = mean(vals);
    end
end
end
