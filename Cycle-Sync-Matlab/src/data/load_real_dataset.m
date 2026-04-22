function data = load_real_dataset(path)
S = load(path);
req = {'AdjMat34','tijMat34','t_orig2_cntrd','R_global'};
for k = 1:numel(req)
    if ~isfield(S,req{k}), error('Missing required field %s', req{k}); end
end
AdjMat = double(S.AdjMat34 ~= 0);
edges = matlab_lower_tri_edges(AdjMat);
gamma = S.tijMat34;
gamma = bsxfun(@rdivide, gamma, max(sqrt(sum(gamma.^2,1)),1e-15));
[~,base,~] = fileparts(path);
ix = strfind(base,'_location');
if ~isempty(ix), name = base(1:ix(1)-1); else, name = base; end
data.name = name;
data.AdjMat = AdjMat;
data.edges = edges;
data.GammaMat = gamma;
data.TMat_gt = S.t_orig2_cntrd;
data.R_global = S.R_global;
data.path = path;
data.n = size(AdjMat,1);
data.m = size(edges,1);
end
