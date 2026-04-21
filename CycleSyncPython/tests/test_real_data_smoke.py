from pathlib import Path


def test_real_data_loader_and_robust_alignment_smoke():
    from cyclesync.real_data import load_real_dataset, evaluate_real_locations
    from cyclesync.real_experiments import default_cycle_params
    from cyclesync.cyclesync import cycle_sync_location

    root = Path(__file__).resolve().parents[1]
    mat = root / "data" / "real_precomputed" / "pipes_location(2).mat"
    if not mat.exists():
        return
    data = load_real_dataset(mat)
    params = default_cycle_params(fast=True)
    params.tmax = 3
    res = cycle_sync_location(data.adj, data.edges, data.gamma, params)
    err = evaluate_real_locations(res.t, data, alignment="robust")
    assert err["median"] < 0.2
    assert err["alignment_inlier_fraction"] >= 0.5
