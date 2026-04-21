from cyclesync.experiments import ExperimentConfig, run_methods


def test_all_methods_smoke():
    cfg = ExperimentConfig(n=10, p=0.75, q=0.2, sigma=0.01, seed=7, run_fused_ta=True, fast=True)
    df, _ = run_methods(cfg)
    assert set(df["method"]) == {"Cycle-Sync", "LUD", "ShapeFit", "BATA", "FusedTA"}
    assert df["median_error"].notna().all()
    assert (df["runtime_sec"] >= 0).all()


def test_cyclesync_default_parameters():
    from cyclesync.cyclesync import CycleSyncParams
    p = CycleSyncParams()
    assert p.tmax == 20
    assert p.beta == 20.0
    assert p.delta == 1e-8
    assert p.loss_a == 4.0
    assert p.lambda_offset == 10.0
    assert p.init_weight_scale == 20.0
