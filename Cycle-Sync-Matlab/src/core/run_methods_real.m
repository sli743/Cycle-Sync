function summary = run_methods_real(paths,alignment,fast)
if nargin < 2 || isempty(alignment), alignment = 'robust'; end
if nargin < 3 || isempty(fast), fast = true; end
all_rows = {};
for p = 1:numel(paths)
    data = load_real_dataset(paths{p});
    methods = {'Cycle-Sync','LUD','ShapeFit','BATA','FusedTA'};
    for a = 1:numel(methods)
        method = methods{a};
        try
            switch method
                case 'Cycle-Sync'
                    if fast, params = default_cycle_params_fast(); else, params = default_cycle_params(); end
                    params.seed = 123;
                    res = cycle_sync_location(data.AdjMat,data.edges,data.GammaMat,params);
                case 'LUD'
                    opts.maxit = 20; opts.delt = 1e-16;
                    res = lud_location(data.AdjMat,data.edges,data.GammaMat,opts);
                case 'ShapeFit'
                    res = shapefit_location(data.edges,data.GammaMat,data.n,struct());
                case 'BATA'
                    opts.seed = 130; res = bata_location(data.edges,data.GammaMat,data.n,opts);
                case 'FusedTA'
                    opts.seed = 136;
                    if fast, opts.numofiterinit = 5; opts.numofouteriter = 3; opts.numofinneriter = 2; end
                    res = fused_ta_location(data.edges,data.GammaMat,data.n,opts);
            end
            err = evaluate_real_locations(res.t,data,alignment,true);
            med = err.median; mn = err.mean; tm = err.trimmed_mean; q25 = err.q25; q75 = err.q75; q90 = err.q90; rt = res.runtime; msg = "";
        catch ME
            med = NaN; mn = NaN; tm = NaN; q25 = NaN; q75 = NaN; q90 = NaN; rt = NaN; msg = string(ME.message);
        end
        all_rows(end+1,:) = {string(data.name), string(method), med, mn, tm, q25, q75, q90, string(alignment), data.n, data.m, rt, msg}; %#ok<AGROW>
    end
end
summary = cell2table(all_rows, 'VariableNames', {'dataset','method','median_error','mean_error','trimmed_mean_error','q25_error','q75_error','q90_error','alignment','n','m','runtime_sec','error'});
end
