"""DiD и placebo-тест проверяем на симуляции, где истинный эффект известен."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regulation_impact import did_estimate, placebo  # noqa: E402


def panel(effect: float, n: int = 12_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 270, n), unit="D")
    above = rng.integers(0, 2, n)
    post = (ts >= pd.Timestamp("2026-04-01")).astype(int)
    p = 0.30 + 0.04 * above + effect * above * post
    return pd.DataFrame(dict(user_id=rng.integers(0, 4000, n), ts=ts, above_cap=above, post=post,
                             category=rng.choice(["a", "b", "c"], n),
                             converted=rng.binomial(1, 0.8, n), bnpl=rng.binomial(1, p)))


def test_did_recovers_known_effect():
    est = did_estimate(panel(-0.20), "bnpl")
    assert est["ci95"][0] <= -0.20 <= est["ci95"][1]
    assert est["p_value"] < 1e-6


def test_trend_adjusted_estimate_close_when_trends_parallel():
    df = panel(-0.20)
    assert abs(did_estimate(df, "bnpl", trends=True)["did"] - did_estimate(df, "bnpl")["did"]) < 0.05


def test_placebo_passes_without_pre_period_effect():
    assert placebo(panel(-0.20), "2026-02-15")["passed"]
