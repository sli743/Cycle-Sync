function result = fused_ta_location(edges,gamma,n,opts)
if nargin < 4, opts = struct(); end
if ~isfield(opts,'seed'), opts.seed = 123; end
rng(opts.seed);
param.delta = getfield_with_default(opts,'delta',1e-5);
param.numofiterinit = getfield_with_default(opts,'numofiterinit',50);
param.relerrthreinit = getfield_with_default(opts,'relerrthreinit',1e-5);
param.numofouteriter = getfield_with_default(opts,'numofouteriter',100);
param.numofinneriter = getfield_with_default(opts,'numofinneriter',5);
param.robustthreRLUD = getfield_with_default(opts,'robustthreRLUD',1e-1);
param.robustthreBATA = getfield_with_default(opts,'robustthreBATA',1e-1);
param.relerrthreouter = getfield_with_default(opts,'relerrthreouter',1e-6);
param.relchangethreouter = getfield_with_default(opts,'relchangethreouter',1e-5);
param.relchangethreinner = getfield_with_default(opts,'relchangethreinner',1e-3);
tic;
tij_index = edges';
tij_observe = -gamma;
[t,ed_ret_idx] = Fused_TA(tij_index,tij_observe,param);
result.t = t;
result.runtime = toc;
result.info.method = 'FusedTA';
result.info.retained_edges = nnz(ed_ret_idx);
end
