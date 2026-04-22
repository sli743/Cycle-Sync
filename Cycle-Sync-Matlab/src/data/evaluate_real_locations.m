function err = evaluate_real_locations(t_est,data,alignment,apply_rotation)
if nargin < 3 || isempty(alignment), alignment = 'robust'; end
if nargin < 4 || isempty(apply_rotation), apply_rotation = true; end
if apply_rotation
    t_eval = data.R_global * t_est;
else
    t_eval = t_est;
end
err = camera_errors(t_eval,data.TMat_gt,alignment);
end
