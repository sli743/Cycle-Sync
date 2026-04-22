root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'src')));
addpath(fullfile(root, 'demo'));
addpath(fullfile(root, 'experiments'));
fprintf('CycleSync MATLAB paths added.\n');
