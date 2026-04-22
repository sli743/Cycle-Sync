function sol = solve_translation_wls(edges,gamma,weights,n,opts)
if nargin < 5, opts = struct(); end
if ~isfield(opts,'alpha_lower'), opts.alpha_lower = 1; end
if ~isfield(opts,'max_iter'), opts.max_iter = 50; end
if ~isfield(opts,'tol'), opts.tol = 1e-10; end
m = size(edges,1);
active = true(m,1);
last = [];
t = zeros(3,n);
status = 0;
for it = 1:opts.max_iter
    H = zeros(3*n,3*n);
    b = zeros(3*n,1);
    I3 = eye(3);
    for e = 1:m
        w = weights(e);
        if w <= 0 || ~isfinite(w), continue; end
        g = gamma(:,e);
        if active(e)
            P = I3; rhs = g;
        else
            P = I3 - g*g'; rhs = zeros(3,1);
        end
        i = edges(e,1); j = edges(e,2);
        ii = (3*i-2):(3*i); jj = (3*j-2):(3*j);
        H(ii,ii) = H(ii,ii) + w*P;
        H(jj,jj) = H(jj,jj) + w*P;
        H(ii,jj) = H(ii,jj) - w*P;
        H(jj,ii) = H(jj,ii) - w*P;
        if active(e)
            b(ii) = b(ii) + w*rhs;
            b(jj) = b(jj) - w*rhs;
        end
    end
    Aeq = kron(ones(1,n), eye(3));
    K = [H + 1e-12*eye(3*n), Aeq'; Aeq, zeros(3,3)];
    x = K \ [b; zeros(3,1)];
    x = x(1:3*n);
    t = reshape(x,3,n);
    xij = t(:,edges(:,1)) - t(:,edges(:,2));
    alpha_star = sum(gamma.*xij,1)';
    new_active = alpha_star <= opts.alpha_lower + opts.tol;
    if ~isempty(last) && isequal(new_active, active)
        status = 1;
        active = new_active;
        break;
    end
    last = active;
    active = new_active;
end
xij = t(:,edges(:,1)) - t(:,edges(:,2));
alpha = max(opts.alpha_lower, sum(gamma.*xij,1));
residual_vec = xij - gamma.*alpha;
residual_norms = sqrt(sum(residual_vec.^2,1));
sol.t = t;
sol.alpha = alpha;
sol.residual_vec = residual_vec;
sol.residual_norms = residual_norms;
sol.status = status;
sol.cost = sum(weights(:)'.*residual_norms.^2);
end
