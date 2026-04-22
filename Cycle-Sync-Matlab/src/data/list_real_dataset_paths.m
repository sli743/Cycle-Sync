function paths = list_real_dataset_paths(data_dir)
if nargin < 1 || isempty(data_dir), data_dir = fullfile('data','real_precomputed'); end
D = dir(fullfile(data_dir,'*_location*.mat'));
paths = cell(numel(D),1);
for k = 1:numel(D)
    paths{k} = fullfile(D(k).folder,D(k).name);
end
paths = sort(paths);
end
