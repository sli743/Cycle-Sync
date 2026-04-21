function X = normalize_cols(X)
n = sqrt(sum(X.^2,1));
X = bsxfun(@rdivide, X, max(n, 1e-15));
end
