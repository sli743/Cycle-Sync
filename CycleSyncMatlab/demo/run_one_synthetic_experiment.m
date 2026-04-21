clear; clc;
startup;
cfg.model = 'uniform';
cfg.n = 100;
cfg.p = 0.5;
cfg.q = 0.8;
cfg.sigma = 0;
cfg.seed = 2025;
cfg.alignment = 'robust';
cfg.fast = true;
cfg.run_fused_ta = true;
[summary, ~] = run_methods_synthetic(cfg);
disp(summary(:,{'method','median_error','mean_error','trimmed_mean_error','runtime_sec'}));
out_dir = fullfile('results','demo_synthetic');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
writetable(summary, fullfile(out_dir,'summary.csv'));
fprintf('Saved %s\n', fullfile(out_dir,'summary.csv'));
