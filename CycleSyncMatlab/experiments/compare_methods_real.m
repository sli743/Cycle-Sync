clear; clc;
startup;
data_dir = fullfile('data','real_precomputed');
paths = list_real_dataset_paths(data_dir);
summary = run_methods_real(paths,'robust',true);
out_dir = fullfile('results','real_method_comparison');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
writetable(summary, fullfile(out_dir,'real_method_by_dataset.csv'));
G = groupsummary(summary,'method',{'mean','median'},{'median_error','mean_error','runtime_sec'});
disp(G);
writetable(G, fullfile(out_dir,'real_method_summary.csv'));
fprintf('Saved results to %s\n', out_dir);
