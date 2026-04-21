function [summary, outputs] = run_methods_synthetic(cfg)
if ~isfield(cfg,'model'), cfg.model = 'uniform'; end
if ~isfield(cfg,'n'), cfg.n = 100; end
if ~isfield(cfg,'p'), cfg.p = 0.5; end
if ~isfield(cfg,'q'), cfg.q = 0.8; end
if ~isfield(cfg,'sigma'), cfg.sigma = 0; end
if ~isfield(cfg,'seed'), cfg.seed = 2025; end
if ~isfield(cfg,'alignment'), cfg.alignment = 'robust'; end
if ~isfield(cfg,'run_fused_ta'), cfg.run_fused_ta = true; end
if ~isfield(cfg,'fast'), cfg.fast = false; end
if strcmpi(cfg.model,'uniform')
    data = uniform_corruption_model(cfg.n,cfg.p,cfg.q,cfg.sigma,cfg.seed);
else
    data = adversarial_corruption_model(cfg.n,cfg.p,cfg.q,cfg.sigma,cfg.seed);
end
methods = {'Cycle-Sync','LUD','ShapeFit','BATA'};
if cfg.run_fused_ta, methods{end+1} = 'FusedTA'; end
rows = cell(numel(methods),1);
outputs.data = data;
for a = 1:numel(methods)
    method = methods{a};
    switch method
        case 'Cycle-Sync'
            if cfg.fast, params = default_cycle_params_fast(); else, params = default_cycle_params(); end
            params.seed = cfg.seed + 1000;
            res = cycle_sync_location(data.AdjMat,data.edges,data.GammaMat,params);
            t_est = res.t; runtime = res.runtime; outputs.CycleSync = res;
        case 'LUD'
            opts.maxit = 20; opts.delt = 1e-16;
            res = lud_location(data.AdjMat,data.edges,data.GammaMat,opts);
            t_est = res.t; runtime = res.runtime; outputs.LUD = res;
        case 'ShapeFit'
            res = shapefit_location(data.edges,data.GammaMat,cfg.n,struct());
            t_est = res.t; runtime = res.runtime; outputs.ShapeFit = res;
        case 'BATA'
            opts.seed = cfg.seed + 2000;
            res = bata_location(data.edges,data.GammaMat,cfg.n,opts);
            t_est = res.t; runtime = res.runtime; outputs.BATA = res;
        case 'FusedTA'
            opts.seed = cfg.seed + 3000;
            if cfg.fast
                opts.numofiterinit = 5; opts.numofouteriter = 3; opts.numofinneriter = 2;
            end
            res = fused_ta_location(data.edges,data.GammaMat,cfg.n,opts);
            t_est = res.t; runtime = res.runtime; outputs.FusedTA = res;
    end
    err = camera_errors(t_est,data.TMat_gt,cfg.alignment);
    rows{a} = {method, err.median, err.mean, err.trimmed_mean, err.q25, err.q75, err.nrmse, runtime, cfg.n, cfg.p, cfg.q, cfg.sigma, string(cfg.model), string(cfg.alignment)};
end
summary = cell2table(vertcat(rows{:}), 'VariableNames', {'method','median_error','mean_error','trimmed_mean_error','q25_error','q75_error','nrmse','runtime_sec','n','p','q','sigma','model','alignment'});
end
