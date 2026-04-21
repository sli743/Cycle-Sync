from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.io as sio
from .align import camera_errors

@dataclass
class RealDataset:
    name: str
    adj: np.ndarray
    edges: np.ndarray
    gamma: np.ndarray
    t_gt: np.ndarray
    R_global: np.ndarray
    path: Path

    @property
    def n(self) -> int:
        return int(self.adj.shape[0])

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

def matlab_lower_tri_edges(adj: np.ndarray) -> np.ndarray:
    """Return edge order compatible with MATLAB [row,col]=find(tril(A,-1)).

    MATLAB scans arrays column-major. The uploaded tijMat34 columns follow this
    ordering. Each returned edge is (i,j) with i<j and gamma[:,e] intended to
    represent t_i - t_j, matching the synthetic MATLAB generators.
    """
    n = adj.shape[0]
    return np.array([(c, r) for c in range(n) for r in range(c + 1, n) if adj[r, c] != 0], dtype=int)

def load_real_dataset(path: str | Path) -> RealDataset:
    path = Path(path)
    mat = sio.loadmat(path)
    required = ["AdjMat34", "tijMat34", "t_orig2_cntrd", "R_global"]
    missing = [k for k in required if k not in mat]
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")
    adj_raw = mat["AdjMat34"]
    adj = adj_raw.toarray() if hasattr(adj_raw, "toarray") else np.asarray(adj_raw)
    adj = (adj != 0).astype(float)
    edges = matlab_lower_tri_edges(adj)
    gamma = np.asarray(mat["tijMat34"], dtype=float)
    if gamma.shape[1] != edges.shape[0]:
        raise ValueError(f"{path.name}: tijMat34 has {gamma.shape[1]} columns but AdjMat34 implies {edges.shape[0]} MATLAB-order edges")
    gamma = gamma / np.maximum(np.linalg.norm(gamma, axis=0, keepdims=True), 1e-15)
    name = path.name.split("_location")[0]
    return RealDataset(name=name, adj=adj, edges=edges, gamma=gamma,
                       t_gt=np.asarray(mat["t_orig2_cntrd"], dtype=float),
                       R_global=np.asarray(mat["R_global"], dtype=float), path=path)

def list_real_dataset_paths(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.glob("*_location*.mat"))

def evaluate_real_locations(t_est: np.ndarray, data: RealDataset, *, alignment: str = "robust", apply_rotation: bool = True, trim_fraction: float = 0.80) -> dict:
    """Evaluate locations with the real-data convention used by apply_baseline_new2.m.

    The MATLAB pipeline first rotates the estimated locations by R_global and
    then removes signed scale and translation with L1 alignment.
    """
    t_eval = data.R_global @ t_est if apply_rotation else t_est
    return camera_errors(t_eval, data.t_gt, method=alignment, trim_fraction=trim_fraction)
