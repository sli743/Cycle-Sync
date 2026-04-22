function edges = matlab_lower_tri_edges(AdjMat)
[rows,cols] = find(tril(AdjMat,-1));
edges = [cols(:), rows(:)];
end
