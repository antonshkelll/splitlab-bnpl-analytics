"""
Сколько стоило отключение рассрочки для корзин дороже 50 000 ₽ (с 1 апреля 2026)?

По 283-ФЗ с 1 апреля 2026 сведения о рассрочках выше 50 000 ₽ передаются в бюро кредитных историй.
В данных заложен сценарий, где сервис в ответ перестал предлагать рассрочку для таких корзин.

Наивное «было/стало» не годится: сервис растёт, и общий BNPL GMV после апреля всё равно вырос.
Поэтому используем difference-in-differences:
  * «обработанная» группа — корзины > 50 000 ₽ (им перестали предлагать рассрочку);
  * контроль — корзины 20–50 000 ₽ (похожие покупки, изменение их не затронуло).
Корзины дороже 150 000 ₽ не берём: их мало, и они не похожи на контрольную группу.
Метрика — доля чекаутов, завершившихся покупкой (любым способом), и доля BNPL.
Плюс оценка потерянного BNPL GMV в рублях.

Проверка допущения о параллельных трендах — placebo-тест: та же регрессия только на данных
до 1 апреля с фиктивной датой «изменения» 15 февраля. Если тренды параллельны, «эффект»
фиктивной даты должен быть близок к нулю и статистически незначим. Дополнительно — DiD
с отдельным линейным трендом для каждой группы: если оценка от этого почти не меняется,
вывод устойчив к расхождению трендов.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import COLORS, FIG, LAW_DATE, ROOT, connect, plt

SQL = """
WITH chk AS (
    SELECT user_id, ts, amount, category
    FROM events WHERE event = 'checkout_start' AND amount BETWEEN 20000 AND 150000
),
paid AS (
    SELECT o.user_id, o.ts, o.amount, o.payment_method FROM orders o
)
SELECT c.user_id, c.ts, c.amount, c.category,
       date_trunc('week', c.ts)::DATE AS week,
       (c.amount > 50000)::INT                    AS above_cap,
       (c.ts >= TIMESTAMP '2026-04-01')::INT       AS post,
       (p.user_id IS NOT NULL)::INT                AS converted,
       coalesce(p.payment_method = 'bnpl', FALSE)::INT AS bnpl
FROM chk c
LEFT JOIN paid p ON p.user_id = c.user_id AND p.amount = c.amount
                AND p.ts BETWEEN c.ts AND c.ts + INTERVAL 5 MINUTE
"""


PLACEBO_DATES = ("2026-02-01", "2026-02-15", "2026-03-01")
ALPHA = 0.05 / 2          # поправка Бонферрони: проверяем две метрики


def did_estimate(df: pd.DataFrame, y: str, post: str = "post", trends: bool = False) -> dict:
    """DiD-оценка эффекта above_cap × post с эффектами категорий и кластеризацией по пользователю.
    trends=True добавляет каждой группе свой линейный тренд по неделям."""
    extra = " + t + above_cap:t" if trends else ""
    if trends and "t" not in df:
        df = df.assign(t=(df.ts - df.ts.min()).dt.days / 7)
    m = smf.ols(f"{y} ~ above_cap * {post}{extra} + C(category)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df.user_id})
    term = f"above_cap:{post}"
    b, ci = m.params[term], m.conf_int().loc[term]
    return dict(did=round(b, 4), ci95=[round(ci[0], 4), round(ci[1], 4)], p_value=float(f"{m.pvalues[term]:.2e}"))


def placebo(df: pd.DataFrame, fake_date: str) -> dict:
    """Та же DiD только на данных до закона, с фиктивной датой изменения."""
    pre = df[df.ts < pd.Timestamp(LAW_DATE)].copy()
    pre["fake_post"] = (pre.ts >= pd.Timestamp(fake_date)).astype(int)
    out = {y: did_estimate(pre, y, post="fake_post") for y in ("converted", "bnpl")}
    out["fake_date"] = fake_date
    out["n_checkouts"] = len(pre)
    out["passed"] = all(out[y]["p_value"] >= ALPHA for y in ("converted", "bnpl"))
    return out


def main() -> dict:
    con = connect()
    df = con.execute(SQL).df()
    df = df[df.ts < "2026-10-01"]

    # DiD-регрессия с кластеризацией ошибок по пользователю
    res = {y: did_estimate(df, y) for y in ("converted", "bnpl")}
    placebo_res = [placebo(df, d) for d in PLACEBO_DATES]
    res_trend = {y: did_estimate(df, y, trends=True) for y in ("converted", "bnpl")}

    g = df.groupby(["above_cap", "post"])[["converted", "bnpl"]].mean().round(4)

    # Потери считаем из тех же DiD-оценок: умножаем эффект на оборот корзин > 50k после изменения.
    # Наивное «было в неделю × недели» не годится — сервис растёт, корзин стало больше.
    weeks_post = df[df.post == 1].week.nunique()
    exposed_gmv = float(df[(df.post == 1) & (df.above_cap == 1)].amount.sum())
    lost_bnpl_gmv = -res["bnpl"]["did"] * exposed_gmv          # оборот, который ушёл из рассрочки
    lost_orders_gmv = -res["converted"]["did"] * exposed_gmv   # оборот, который маркетплейс потерял совсем
    switched_share = 1 - res["converted"]["did"] / res["bnpl"]["did"]

    # график: конверсия чекаута в покупку по неделям, две группы
    wk = df.groupby(["week", "above_cap"]).agg(conv=("converted", "mean"), bnpl=("bnpl", "mean")).reset_index()
    wk["week"] = pd.to_datetime(wk.week)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, title in [(axes[0], "bnpl", "Доля чекаутов, оплаченных частями"),
                           (axes[1], "conv", "Конверсия чекаута в покупку (любой способ)")]:
        for cap, color, lab in [(0, COLORS["main"], "корзина 20–50 тыс. ₽ (контроль)"),
                                (1, COLORS["accent"], "корзина > 50 тыс. ₽")]:
            s = wk[wk.above_cap == cap].set_index("week")[col].rolling(3, min_periods=1).mean()
            ax.plot(s.index, s * 100, color=color, label=lab, lw=2)
        ax.axvline(pd.Timestamp(LAW_DATE), color="k", ls="--", lw=1)
        ax.set(title=title, ylabel="%")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "regulation_did.png"); plt.close(fig)

    out = dict(
        n_checkouts=len(df),
        group_means={f"above_cap={a},post={p}": v for (a, p), v in g.to_dict("index").items()},
        did=res,
        placebo=placebo_res,
        did_with_group_trends=res_trend,
        weeks_post=int(weeks_post),
        exposed_gmv_rub=round(exposed_gmv),
        lost_bnpl_gmv_rub=round(lost_bnpl_gmv),
        lost_marketplace_gmv_rub=round(lost_orders_gmv),
        switched_to_other_payment_share=round(switched_share, 3),
    )
    (ROOT / "reports" / "regulation_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    main()
