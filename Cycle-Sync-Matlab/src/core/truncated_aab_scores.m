function out = truncated_aab_scores(AdjMat,edges,gamma,params)
if isfield(params,'seed') && ~isempty(params.seed), rng(params.seed); end
n = size(AdjMat,1); m = size(edges,1);
nsample = params.taab_nsample;
G = direction_tensor(gamma,edges,n);
L = edge_lookup(edges,n);
thresh = sqrt(max(0, 1 - params.sinmin^2));
samples = -ones(nsample,m);
has_cycle = false(1,m);
SAAB = (pi/2)*ones(nsample,m);
for e = 1:m
    i = edges(e,1); j = edges(e,2);
    c = find(AdjMat(:,i) > 0 & AdjMat(:,j) > 0);
    c = c(c ~= i & c ~= j);
    if ~isempty(c)
        a = squeeze(G(:,i,c));
        b = squeeze(G(:,j,c));
        if isvector(a), a = a(:); b = b(:); end
        cosv = abs(sum(a.*b,1));
        good = c(cosv < thresh);
        if ~isempty(good), c = good; end
        picks = c(randi(numel(c), nsample, 1));
        samples(:,e) = picks;
        has_cycle(e) = true;
        gij = gamma(:,e);
        for s = 1:nsample
            k = picks(s);
            Xki = G(:,i,k);
            Xjk = -G(:,j,k);
            X = dot(Xki,gij);
            Y = dot(Xjk,gij);
            Z = dot(Xki,Xjk);
            S = double((X < Y*Z) && (Y < X*Z));
            den = 1 - Z^2;
            if abs(den) < 1e-12, den = sign(den + eps)*1e-12; end
            arg = S*(X^2 + Y^2 - 2*X*Y*Z)/den + (S-1)*min(X,Y);
            SAAB(s,e) = abs(acos(max(min(arg,1),-1)));
        end
    end
end
scores = mean(SAAB,1);
tau = 1; tau_rate = 2; tau_max = 20;
for iter = 1:params.taab_iters
    tau = min(tau*tau_rate, tau_max);
    new_scores = scores;
    for e = 1:m
        if ~has_cycle(e), continue; end
        i = edges(e,1); j = edges(e,2);
        ks = samples(:,e);
        ik = abs(L(i,ks)); jk = abs(L(j,ks));
        w = exp(-tau*(scores(ik) + scores(jk))');
        sw = sum(w);
        if sw <= 1e-300 || ~isfinite(sw)
            w = ones(nsample,1)/nsample;
        else
            w = w/sw;
        end
        new_scores(e) = sum(w.*SAAB(:,e));
    end
    scores = new_scores;
end
if params.normalize_taab_by_pi
    scores = scores/pi;
    raw = SAAB/pi;
else
    raw = SAAB;
end
out.scores = scores(:)';
out.samples = samples;
out.has_cycle = has_cycle;
out.raw_cycle_scores = raw;
end
