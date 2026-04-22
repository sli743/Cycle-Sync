clear; clc;
startup;
mat_file = fullfile('data','real_precomputed','delivery_area_location.mat');
data = load_real_dataset(mat_file);
params = default_cycle_params_fast();
res = cycle_sync_location(data.AdjMat,data.edges,data.GammaMat,params);
err = evaluate_real_locations(res.t,data,'robust',true);
fprintf('Dataset: %s  n=%d  m=%d\n', data.name, data.n, data.m);
fprintf('Alignment: robust\n');
fprintf('Median error: %.6g\n', err.median);
fprintf('Mean error:   %.6g\n', err.mean);
fprintf('Runtime:      %.3f sec\n', res.runtime);
out_dir = fullfile('results','demo_real');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
T = table(string(data.name), err.median, err.mean, err.trimmed_mean, res.runtime, 'VariableNames', {'dataset','median_error','mean_error','trimmed_mean_error','runtime_sec'});
writetable(T, fullfile(out_dir,'summary.csv'));
