function result = shapefit_location(edges,gamma,n,opts)
if nargin < 4, opts = struct(); end
tic;
m = size(edges,1);
E = zeros(m,n);
for k = 1:m
    E(k,edges(k,1)) = 1;
    E(k,edges(k,2)) = -1;
end
v = gamma';
[t,resids] = solve_shapefit(E,v,false);
result.t = t';
result.runtime = toc;
result.info.iterations = numel(resids);
result.info.resids = resids;
end
