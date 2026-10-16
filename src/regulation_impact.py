"""
Как лимит 50 000 ₽ из 283-ФЗ (с 1 апреля 2026) повлиял на BNPL?

Наивное «было/стало» не годится: сервис растёт, и общий BNPL GMV после апреля всё равно вырос.
Поэтому используем difference-in-differences:
  * «обработанная» группа — корзины > 50 000 ₽ (им перестали предлагать рассрочку);
  * контроль — корзины 20–50 000 ₽ (похожие покупки, лимит их не затронул).
Метрика — доля чекаутов, завершившихся покупкой (любым способом), и доля BNPL.
Плюс оценка потерянного BNPL GMV в рублях.
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


def main() -> dict:
    con = connect()
    df = con.execute(SQL).df()
    df = df[df.ts < "2026-10-01"]

    # DiD-регрессия с кластеризацией ошибок по пользователю
    res = {}
    for y in ("converted", "bnpl"):
        m = smf.ols(f"{y} ~ above_cap * post + C(category)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df.user_id})
        b, ci = m.params["above_cap:post"], m.conf_int().loc["above_cap:post"]
        res[y] = dict(did=round(b, 4), ci95=[round(ci[0], 4), round(ci[1], 4)], p_value=float(f"{m.pvalues['above_cap:post']:.2e}"))

    g = df.groupby(["above_cap", "post"])[["converted", "bnpl"]].mean().round(4)

    # потерянный BNPL GMV: сколько BNPL-оборота приходилось на корзины > 50k до закона (в неделю)
    weekly_pre = df[(df.post == 0) & (df.above_cap == 1) & (df.bnpl == 1)].groupby("week").amount.sum()
    weeks_post = df[df.post == 1].week.nunique()
    lost_bnpl_gmv = float(weekly_pre.mean() * weeks_post)
    # часть этих покупок всё равно состоялась картой — реально «потерянный» GMV маркетплейса меньше
    lost_orders_gmv = float(-res["converted"]["did"] * df[(df.post == 1) & (df.above_cap == 1)].amount.sum())

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
        weeks_post=int(weeks_post),
        lost_bnpl_gmv_rub=round(lost_bnpl_gmv),
        lost_marketplace_gmv_rub=round(lost_orders_gmv),
    )
    (ROOT / "reports" / "regulation_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    main()
