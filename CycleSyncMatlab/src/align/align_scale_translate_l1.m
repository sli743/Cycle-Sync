function [fit,shift,scale,nrmse] = align_scale_translate_l1(t_est,t_ref)
[~,shift0,scale0,~] = align_scale_translate_l2(t_est,t_ref);
x0 = [shift0(:); scale0];
obj = @(x) sum(sqrt(sum((t_ref - (x(4)*t_est + x(1:3))).^2,1) + 1e-12));
opts = optimset('Display','off','MaxIter',500,'TolX',1e-9,'TolFun',1e-9);
x = fminsearch(obj,x0,opts);
shift = x(1:3);
scale = x(4);
fit = scale*t_est + shift;
Y0 = t_ref - mean(t_ref,2);
nrmse = sqrt(sum(sum((t_ref-fit).^2))/max(sum(sum(Y0.^2)),1e-15));
end
