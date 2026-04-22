function err = camera_errors(t_est,t_ref,method,trim_fraction)
if nargin < 3 || isempty(method), method = 'robust'; end
if nargin < 4 || isempty(trim_fraction), trim_fraction = 0.80; end
switch lower(method)
    case 'l1'
        [fit,shift,scale,nrmse] = align_scale_translate_l1(t_est,t_ref);
        inlier_mask = true(1,size(t_est,2)); weights = ones(1,size(t_est,2));
    case 'l2'
        [fit,shift,scale,nrmse] = align_scale_translate_l2(t_est,t_ref);
        inlier_mask = true(1,size(t_est,2)); weights = ones(1,size(t_est,2));
    otherwise
        [fit,shift,scale,nrmse,inlier_mask,weights] = align_scale_translate_robust(t_est,t_ref,trim_fraction);
end
e = sqrt(sum((fit-t_ref).^2,1));
se = sort(e);
keep = max(1,ceil(trim_fraction*numel(e)));
err.t_fit = fit;
err.shift = shift;
err.scale = scale;
err.nrmse = nrmse;
err.errors = e;
err.median = median(e);
err.mean = mean(e);
err.trimmed_mean = mean(se(1:keep));
err.q25 = quantile(e,0.25);
err.q75 = quantile(e,0.75);
err.q90 = quantile(e,0.90);
err.max = max(e);
err.alignment_inlier_mask = inlier_mask;
err.alignment_weights = weights;
err.alignment_inlier_fraction = mean(inlier_mask);
end
