"""Статистические методы проверяем симуляциями, где истина известна."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ab_test import cuped, mde, peeking_simulation, prop_test, ratio_delta, srm_check  # noqa: E402

rng = np.random.default_rng(42)


def test_srm_detects_broken_split():
    assert srm_check(10_000, 10_050)["ok"]
    assert not srm_check(10_000, 10_600)["ok"]


def test_prop_test_ci_covers_true_effect():
    hits = 0
    for _ in range(300):
        c, t = rng.binomial(1, 0.05, 8000), rng.binomial(1, 0.06, 8000)
        lo, hi = prop_test(c.sum(), len(c), t.sum(), len(t))["ci95_abs"]
        hits += lo <= 0.01 <= hi
    assert 0.92 <= hits / 300 <= 0.98


def test_cuped_reduces_variance_without_bias():
    n = 20_000
    x = rng.gamma(2, 50, n)                      # метрика до теста
    group = rng.integers(0, 2, n)
    y = 0.8 * x + rng.normal(0, 20, n) + 5 * group
    df = pd.DataFrame({"y": y, "x": x})
    adj, _ = cuped(df, "y", ["x"])
    raw_eff = y[group == 1].mean() - y[group == 0].mean()
    adj_eff = adj[group == 1].mean() - adj[group == 0].mean()
    assert abs(adj_eff - 5) < 1.0 and abs(raw_eff - adj_eff) < 3
    assert adj.var() < 0.2 * y.var()


def test_delta_method_coverage_for_ratio_metric():
    """Пользователь берёт несколько займов, просрочки коррелированы внутри пользователя."""
    hits = 0
    for _ in range(200):
        def users(n):
            loans = rng.poisson(1.5, n) + 1
            risk = rng.beta(1, 30, n)
            return rng.binomial(loans, risk).astype(float), loans.astype(float)
        nc, dc = users(3000)
        nt, dt = users(3000)
        lo, hi = ratio_delta(nc, dc, nt, dt)["ci95_abs"]
        hits += lo <= 0 <= hi
    assert 0.91 <= hits / 200 <= 0.99


def test_mde_shrinks_with_sample():
    assert mde(0.06, 40_000) < mde(0.06, 10_000)


def test_peeking_inflates_false_positives():
    sim, _ = peeking_simulation(sims=1000)
    assert sim["fpr_fixed_horizon"] < 0.08
    assert sim["fpr_daily_peeking"] > 0.15
