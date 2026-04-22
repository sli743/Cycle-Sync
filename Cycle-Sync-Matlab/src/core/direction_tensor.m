function G = direction_tensor(gamma,edges,n)
G = zeros(3,n,n);
for e = 1:size(edges,1)
    i = edges(e,1); j = edges(e,2);
    G(:,j,i) = gamma(:,e);
    G(:,i,j) = -gamma(:,e);
end
end
