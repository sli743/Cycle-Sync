function data = uniform_corruption_model(n,p,q,sigma,seed,dist)
if nargin < 5 || isempty(seed), seed = 2025; end
if nargin < 6 || isempty(dist), dist = 'uniform'; end
rng(seed);
G = rand(n,n) < p;
G = tril(G,-1);
AdjMat = G + G';
[Ind_j, Ind_i] = find(G == 1);
m = numel(Ind_i);
TMat_gt = randn(3,n);
GammaMat_gt = TMat_gt(:,Ind_i) - TMat_gt(:,Ind_j);
edge_len = sqrt(sum(GammaMat_gt.^2,1));
GammaMat_gt = bsxfun(@rdivide, GammaMat_gt, max(edge_len,1e-15));
GammaMat = GammaMat_gt;
noiseInd = rand(1,m) >= q;
corrInd = ~noiseInd;
noise = randn(3,m);
if strcmpi(dist,'uniform')
    noise = normalize_cols(noise);
end
GammaMat(:,noiseInd) = GammaMat_gt(:,noiseInd) + sigma*noise(:,noiseInd);
GammaMat(:,corrInd) = noise(:,corrInd);
GammaMat = normalize_cols(GammaMat);
true_error = abs(acos(max(min(sum(GammaMat_gt.*GammaMat,1),1),-1)));
data.AdjMat = AdjMat;
data.edges = [Ind_i(:), Ind_j(:)];
data.GammaMat = GammaMat;
data.GammaMat_gt = GammaMat_gt;
data.TMat_gt = TMat_gt;
data.true_error = true_error;
data.edge_len = edge_len;
data.corrupted = corrInd;
data.model = 'uniform';
end
