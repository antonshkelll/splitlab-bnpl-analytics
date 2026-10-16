"""
A/B-тест «Сплит на карточке товара» (3–30 августа 2026).

Гипотеза: если показать «4 × N ₽» прямо на карточке товара, больше людей выберут оплату
частями, и вырастет BNPL GMV на пользователя. Риск: импульсные покупки → выше просрочка.

Дизайн:
  * рандомизация по user_id (хеш с солью), 50/50, в анализ попадают пользователи с визитом в период теста;
  * основная метрика — конверсия пользователя в BNPL-покупку;
  * вторичная — BNPL GMV на пользователя (с CUPED; ковариаты — история за 6 месяцев до теста);
  * guardrails — общая конверсия в покупку (нет ли каннибализации) и FPD30 (первый платёж
    просрочен на 30+ дней; ratio-метрика → дельта-метод).
Проверки: SRM, мощность/MDE заранее, A/A-симуляция «подглядывания».
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import COLORS, FIG, ROOT, connect, plt

AB_START, AB_END = "2026-08-03", "2026-08-31"
PRE_START, PRE_END = "2026-02-02", "2026-08-03"   # 6 месяцев до теста — для CUPED
rng = np.random.default_rng(0)

USER_SQL = f"""
WITH exp_users AS (
    SELECT DISTINCT e.user_id FROM events e
    WHERE e.event = 'session_start' AND e.ts >= '{AB_START}' AND e.ts < '{AB_END}'
),
m AS (
    SELECT user_id,
           count(*) FILTER (WHERE payment_method = 'bnpl')                    AS bnpl_orders,
           coalesce(sum(amount) FILTER (WHERE payment_method = 'bnpl'), 0)    AS bnpl_gmv,
           count(*)                                                           AS orders
    FROM orders WHERE ts >= '{AB_START}' AND ts < '{AB_END}' GROUP BY 1
),
pre AS (
    SELECT user_id,
           coalesce(sum(amount) FILTER (WHERE payment_method = 'bnpl'), 0) AS bnpl_gmv_pre,
           count(*) FILTER (WHERE payment_method = 'bnpl')                  AS bnpl_orders_pre,
           count(*)                                                         AS orders_pre
    FROM orders WHERE ts >= '{PRE_START}' AND ts < '{PRE_END}' GROUP BY 1
),
pre_sessions AS (
    SELECT user_id, count(*) AS sessions_pre FROM events
    WHERE event = 'session_start' AND ts >= '{PRE_START}' AND ts < '{PRE_END}' GROUP BY 1
),
fpd AS (   -- FPD30: второй платёж (первый после покупки) не оплачен в течение 30 дней
    SELECT o.user_id,
           count(*)                                                                  AS bnpl_orders_matured,
           count(*) FILTER (WHERE i.paid_date IS NULL OR i.paid_date > i.due_date + INTERVAL 30 DAY) AS fpd30
    FROM orders o JOIN installments i ON i.order_id = o.order_id AND i.part_no = 2
    WHERE o.ts >= '{AB_START}' AND o.ts < '{AB_END}'
    GROUP BY 1
)
SELECT u.user_id, a."group",
       coalesce(m.bnpl_orders, 0) > 0     AS bnpl_converted,
       coalesce(m.orders, 0) > 0          AS any_converted,
       coalesce(m.bnpl_gmv, 0)            AS bnpl_gmv,
       coalesce(p.bnpl_gmv_pre, 0)        AS bnpl_gmv_pre,
       coalesce(p.bnpl_orders_pre, 0)     AS bnpl_orders_pre,
       coalesce(p.orders_pre, 0)          AS orders_pre,
       coalesce(s.sessions_pre, 0)        AS sessions_pre,
       coalesce(f.bnpl_orders_matured, 0) AS n_loans,
       coalesce(f.fpd30, 0)               AS n_fpd30
FROM exp_users u
JOIN ab_assignments a USING (user_id)
LEFT JOIN m USING (user_id)
LEFT JOIN pre p USING (user_id)
LEFT JOIN pre_sessions s USING (user_id)
LEFT JOIN fpd f USING (user_id)
"""


def srm_check(n_c, n_t):
    chi2, p = stats.chisquare([n_c, n_t])
    return dict(control=int(n_c), treatment=int(n_t), p_value=round(float(p), 4), ok=bool(p > 0.001))


def prop_test(x_c, n_c, x_t, n_t):
    p_c, p_t = x_c / n_c, x_t / n_t
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    d = p_t - p_c
    z = d / se
    return dict(control=round(p_c, 4), treatment=round(p_t, 4), abs_diff=round(d, 4),
                rel_lift=round(d / p_c, 4), ci95_abs=[round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)],
                p_value=round(float(2 * stats.norm.sf(abs(z))), 4))


def mean_test(c, t):
    d = t.mean() - c.mean()
    se = np.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t))
    return dict(control=round(c.mean(), 2), treatment=round(t.mean(), 2), abs_diff=round(d, 2),
                rel_lift=round(d / c.mean(), 4), ci95_abs=[round(d - 1.96 * se, 2), round(d + 1.96 * se, 2)],
                se=round(se, 3), p_value=round(float(2 * stats.norm.sf(abs(d / se))), 4))


def cuped(df, y, xs):
    """CUPED с несколькими ковариатами: y' = y − (X − X̄)θ, θ — МНК на объединённой выборке.
    Ковариаты измерены ДО теста, поэтому не зависят от группы и не смещают оценку эффекта."""
    X = df[xs].to_numpy(float); X = X - X.mean(0)
    theta, *_ = np.linalg.lstsq(X, df[y].to_numpy(float) - df[y].mean(), rcond=None)
    return df[y] - X @ theta, dict(zip(xs, np.round(theta, 3)))


def ratio_delta(num_c, den_c, num_t, den_t):
    """Дельта-метод для ratio-метрики (FPD30 = Σ просрочек / Σ займов), единица рандомизации — пользователь."""
    def mv(num, den):
        n = len(num); mn, md = num.mean(), den.mean()
        r = mn / md
        var = (num.var(ddof=1) / md**2 - 2 * r * np.cov(num, den)[0, 1] / md**2 + r**2 * den.var(ddof=1) / md**2) / n
        return r, var
    r_c, v_c = mv(num_c, den_c); r_t, v_t = mv(num_t, den_t)
    d, se = r_t - r_c, np.sqrt(v_c + v_t)
    return dict(control=round(r_c, 4), treatment=round(r_t, 4), abs_diff=round(d, 4), rel_lift=round(d / r_c, 4),
                ci95_abs=[round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)],
                p_value=round(float(2 * stats.norm.sf(abs(d / se))), 4), loans=[int(den_c.sum()), int(den_t.sum())])


def bootstrap_diff(c, t, n_boot=3000):
    c, t = np.asarray(c), np.asarray(t)
    bc = rng.choice(c, (n_boot, len(c))).mean(1)
    bt = rng.choice(t, (n_boot, len(t))).mean(1)
    lo, hi = np.percentile(bt - bc, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def mde(p, n_per_group, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * np.sqrt(2 * p * (1 - p) / n_per_group)


def peeking_simulation(p=0.035, n_per_day=500, days=28, sims=2000):
    """A/A: эффекта нет. Если смотреть p-value каждый день и останавливаться при p<0.05 — сколько ложных побед?"""
    xc = rng.binomial(n_per_day, p, (sims, days)).cumsum(1)
    xt = rng.binomial(n_per_day, p, (sims, days)).cumsum(1)
    n = n_per_day * np.arange(1, days + 1)
    pc, pt = xc / n, xt / n
    pool = (xc + xt) / (2 * n)
    z = (pt - pc) / np.sqrt(pool * (1 - pool) * 2 / n)
    sig = np.abs(z) > 1.96
    fpr_by_looks = [float(sig[:, :k].any(1).mean()) for k in range(1, days + 1)]
    return dict(fpr_fixed_horizon=round(float(sig[:, -1].mean()), 3),
                fpr_daily_peeking=round(fpr_by_looks[-1], 3)), fpr_by_looks


def main() -> dict:
    con = connect()
    df = con.execute(USER_SQL).df()
    c, t = df[df.group == "control"], df[df.group == "treatment"]
    out = dict(users=len(df), srm=srm_check(len(c), len(t)))

    # мощность: какой эффект мы вообще могли увидеть (база — контроль, оценка «на будущее»)
    base = float(c.bnpl_converted.mean())
    out["power"] = dict(baseline_bnpl_conv=round(base, 4),
                        mde_abs=round(float(mde(base, len(df) / 2)), 4),
                        mde_rel=round(float(mde(base, len(df) / 2) / base), 3))

    out["primary_bnpl_conversion"] = prop_test(c.bnpl_converted.sum(), len(c), t.bnpl_converted.sum(), len(t))
    out["guardrail_any_conversion"] = prop_test(c.any_converted.sum(), len(c), t.any_converted.sum(), len(t))

    raw = mean_test(c.bnpl_gmv, t.bnpl_gmv)
    covs = ["bnpl_gmv_pre", "bnpl_orders_pre", "orders_pre", "sessions_pre"]
    df["bnpl_gmv_cuped"], theta = cuped(df, "bnpl_gmv", covs)
    c, t = df[df.group == "control"], df[df.group == "treatment"]
    cup = mean_test(c.bnpl_gmv_cuped, t.bnpl_gmv_cuped)
    corr = df[["bnpl_gmv"] + covs].corr().iloc[0, 1:].round(3).to_dict()
    out["secondary_bnpl_gmv_per_user"] = dict(
        raw=raw, cuped=cup, theta=theta, corr_with_covariates=corr,
        variance_reduction=round(1 - (cup["se"] / raw["se"]) ** 2, 3),
        bootstrap_ci95_raw=bootstrap_diff(c.bnpl_gmv, t.bnpl_gmv))
    # винсоризация: режем хвост по 99-му перцентилю «покупателей» (меняет оцениваемую величину — фиксируем заранее)
    cap = float(df.loc[df.bnpl_gmv > 0, "bnpl_gmv"].quantile(0.99))
    w = mean_test(c.bnpl_gmv.clip(upper=cap), t.bnpl_gmv.clip(upper=cap))
    out["secondary_bnpl_gmv_per_user"]["winsorized_p99"] = dict(cap_rub=round(cap), **w,
        variance_reduction=round(1 - (w["se"] / raw["se"]) ** 2, 3))

    out["guardrail_fpd30"] = ratio_delta(c.n_fpd30.to_numpy(float), c.n_loans.to_numpy(float),
                                         t.n_fpd30.to_numpy(float), t.n_loans.to_numpy(float))

    sim, curve = peeking_simulation()
    out["aa_peeking_simulation"] = sim

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(range(1, len(curve) + 1), np.array(curve) * 100, color=COLORS["accent"], lw=2)
    ax.axhline(5, color="k", ls="--", lw=1)
    ax.text(1, 5.5, "заявленные 5%", fontsize=8)
    ax.set(xlabel="сколько раз «подглядывали» (раз в день)", ylabel="ложноположительные, %",
           title="A/A-тест: подглядывание раздувает ошибку I рода")
    fig.tight_layout(); fig.savefig(FIG / "peeking.png"); plt.close(fig)

    labels = ["Конверсия в BNPL\n(основная)", "BNPL GMV / польз.\n(CUPED)", "Конверсия в покупку\n(guardrail)", "FPD30\n(guardrail)"]
    blocks = [out["primary_bnpl_conversion"], cup, out["guardrail_any_conversion"], out["guardrail_fpd30"]]
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, b in enumerate(blocks):
        base_v = b["control"]
        lo, hi = (np.array(b["ci95_abs"]) / base_v) * 100
        mid = b["rel_lift"] * 100
        color = COLORS["green"] if lo > 0 and i < 3 else (COLORS["accent"] if i == 3 and lo > 0 else COLORS["grey"])
        ax.errorbar(mid, i, xerr=[[mid - lo], [hi - mid]], fmt="o", color=color, capsize=4, lw=2)
        ax.text(hi + 1, i, f"{mid:+.1f}%  (p={b['p_value']:.3f})", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
    ax.set(xlabel="относительный эффект, % (95% ДИ)", title="Итоги A/B «Сплит на карточке товара»")
    ax.set_xlim(right=ax.get_xlim()[1] + 25)
    fig.tight_layout(); fig.savefig(FIG / "ab_results.png"); plt.close(fig)

    (ROOT / "reports" / "ab_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    main()
