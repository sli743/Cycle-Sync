function [fit,shift,scale,nrmse,inlier_mask,weights] = align_scale_translate_robust(t_est,t_ref,trim_fraction,max_iter,cauchy_scale)
if nargin < 3 || isempty(trim_fraction), trim_fraction = 0.80; end
if nargin < 4 || isempty(max_iter), max_iter = 50; end
if nargin < 5 || isempty(cauchy_scale), cauchy_scale = 2.5; end
n = size(t_est,2);
if n <= 4
    [fit,shift,scale,nrmse] = align_scale_translate_l1(t_est,t_ref);
    inlier_mask = true(1,n); weights = ones(1,n); return;
end
[fit,shift,scale,~] = align_scale_translate_l1(t_est,t_ref);
weights = ones(1,n);
min_keep = min(n, max(4, ceil(trim_fraction*n)));
for it = 1:max_iter
    residual = sqrt(sum((t_ref-fit).^2,1));
    med = median(residual);
    madv = 1.4826*median(abs(residual-med));
    sigma = max([madv, med/2, 1e-8]);
    [~,ord] = sort(residual,'ascend');
    keep = false(1,n); keep(ord(1:min_keep)) = true;
    wnew = 1./(1 + (residual/(cauchy_scale*sigma)).^2);
    wnew(~keep) = 0;
    [fit_new,shift,scale,~] = align_scale_translate_l2(t_est,t_ref,wnew);
    if norm(fit_new-fit,'fro')/max(norm(fit,'fro'),1e-12) < 1e-8
        weights = wnew; fit = fit_new; break;
    end
    weights = wnew; fit = fit_new;
end
Y0 = t_ref - mean(t_ref,2);
nrmse = sqrt(sum(sum((t_ref-fit).^2))/max(sum(sum(Y0.^2)),1e-15));
inlier_mask = weights > 0;
end
