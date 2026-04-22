# Precomputed real-data location inputs

These small `.mat` files are precomputed ETH3D-style location-estimation inputs supplied with this test package.  Each file contains:

- `AdjMat34`: viewing graph after preprocessing
- `tijMat34`: pairwise translation directions in MATLAB `find(tril(A,-1))` column-major edge order
- `t_orig2_cntrd`: centered and scaled ground-truth camera locations
- `R_global`: global rotation alignment used before signed-scale/translation location alignment

The Python loader in `cyclesync.real_data` preserves the MATLAB edge ordering.  Using NumPy's row-major `nonzero(tril(A))` order will mismatch the columns of `tijMat34`.
