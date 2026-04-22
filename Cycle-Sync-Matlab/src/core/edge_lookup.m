function L = edge_lookup(edges,n)
L = zeros(n,n);
for e = 1:size(edges,1)
    i = edges(e,1); j = edges(e,2);
    L(i,j) = e;
    L(j,i) = -e;
end
end
