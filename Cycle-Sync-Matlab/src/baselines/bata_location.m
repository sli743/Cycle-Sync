function result = bata_location(edges,gamma,n,opts)
if nargin < 4, opts = struct(); end
if ~isfield(opts,'seed'), opts.seed = 123; end
rng(opts.seed);
param.delta = getfield_with_default(opts,'delta',1e-6);
param.numofiterinit = getfield_with_default(opts,'numofiterinit',10);
param.numofouteriter = getfield_with_default(opts,'numofouteriter',10);
param.numofinneriter = getfield_with_default(opts,'numofinneriter',10);
param.robustthre = getfield_with_default(opts,'robustthre',1e-1);
tic;
tij_index = edges';
tij_observe = -gamma;
t = BATA(tij_index,tij_observe,param);
result.t = t;
result.runtime = toc;
result.info.method = 'BATA';
end
