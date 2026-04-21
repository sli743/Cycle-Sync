function g = oriented_gamma(gamma,L,u,v)
idx = L(u,v);
if idx > 0
    g = gamma(:,idx);
elseif idx < 0
    g = -gamma(:,-idx);
else
    error('Requested missing edge.');
end
end
