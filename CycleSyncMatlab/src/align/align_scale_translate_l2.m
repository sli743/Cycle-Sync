function [fit,shift,scale,nrmse] = align_scale_translate_l2(t_est,t_ref,weights)
if nargin < 3 || isempty(weights)
    weights = ones(1,size(t_est,2));
end
weights = weights(:)'; weights = max(weights,0);
if sum(weights) <= 1e-15, weights = ones(size(weights)); end
weights = weights/sum(weights);
mu_x = t_est*weights';
mu_y = t_ref*weights';
X = t_est - mu_x;
Y = t_ref - mu_y;
denom = sum(weights.*sum(X.^2,1));
if denom < 1e-15
    scale = 0;
else
    scale = sum(weights.*sum(X.*Y,1))/denom;
end
shift = mu_y - scale*mu_x;
fit = scale*t_est + shift;
Y0 = t_ref - mean(t_ref,2);
nrmse = sqrt(sum(sum((t_ref-fit).^2))/max(sum(sum(Y0.^2)),1e-15));
end
