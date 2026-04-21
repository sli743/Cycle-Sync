function data = adversarial_corruption_model(n,p,q,sigma,seed)
if nargin < 5 || isempty(seed), seed = 2025; end
rng(seed);
G = rand(n,n) < p;
G = tril(G,-1);
AdjMat = G + G';
[Ind_j, Ind_i] = find(G == 1);
m = numel(Ind_i);
Tall = randn(3,2*n);
TMat_gt = Tall(:,1:n);
TMat_adv = Tall(:,n+1:end);
GammaMat_gt = TMat_gt(:,Ind_i) - TMat_gt(:,Ind_j);
GammaMat_adv = TMat_adv(:,Ind_i) - TMat_adv(:,Ind_j);
edge_len = sqrt(sum(GammaMat_gt.^2,1));
GammaMat_gt = bsxfun(@rdivide, GammaMat_gt, max(edge_len,1e-15));
GammaMat_adv = normalize_cols(GammaMat_adv);
noiseInd = rand(1,m) >= q;
corrInd = ~noiseInd;
GammaMat = GammaMat_gt;
GammaMat(:,corrInd) = GammaMat_adv(:,corrInd);
noise = normalize_cols(randn(3,m));
GammaMat = normalize_cols(GammaMat + sigma*noise);
true_error = abs(acos(max(min(sum(GammaMat_gt.*GammaMat,1),1),-1)));
data.AdjMat = AdjMat;
data.edges = [Ind_i(:), Ind_j(:)];
data.GammaMat = GammaMat;
data.GammaMat_gt = GammaMat_gt;
data.TMat_gt = TMat_gt;
data.true_error = true_error;
data.edge_len = edge_len;
data.corrupted = corrInd;
data.model = 'adversarial';
end
