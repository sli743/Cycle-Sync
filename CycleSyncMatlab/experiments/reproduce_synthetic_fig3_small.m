clear; clc;
startup;
qlist = [0.2 0.4 0.6 0.8];
trials = 3;
rows = {};
for iq = 1:numel(qlist)
    for r = 1:trials
        cfg.model = 'uniform'; cfg.n = 100; cfg.p = 0.5; cfg.q = qlist(iq); cfg.sigma = 0;
        cfg.seed = 2025 + r - 1; cfg.alignment = 'robust'; cfg.fast = true; cfg.run_fused_ta = true;
        S = run_methods_synthetic(cfg);
        for k = 1:height(S)
            row = table2cell(S(k,:));
            row{end+1} = r - 1;
            rows(end+1,:) = row; %#ok<AGROW>
        end
    end
end
summary = cell2table(rows, 'VariableNames', {'method','median_error','mean_error','trimmed_mean_error','q25_error','q75_error','nrmse','runtime_sec','n','p','q','sigma','model','alignment','trial'});
out_dir = fullfile('results','synthetic_sweep');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
writetable(summary, fullfile(out_dir,'synthetic_sweep_by_trial.csv'));
G = groupsummary(summary, {'method','q'}, {'mean','std'}, {'median_error','runtime_sec'});
disp(G);
writetable(G, fullfile(out_dir,'synthetic_sweep_summary.csv'));
