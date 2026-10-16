"""Строит витрины (CSV для DataLens / Superset) и графики продуктового дашборда."""
import json

import numpy as np
import pandas as pd

from common import AS_OF, COLORS, ECON, FIG, MARTS, ROOT, connect, plt, run_sql


def main() -> dict:
    con = connect()
    daily = run_sql(con, "01_daily_metrics.sql")
    funnel = run_sql(con, "02_funnel.sql")
    cohorts = run_sql(con, "03_cohorts.sql")
    econ = run_sql(con, "04_unit_economics.sql", {**ECON, "as_of": AS_OF})
    for name, df in dict(daily_metrics=daily, funnel=funnel, cohorts=cohorts, unit_economics=econ).items():
        df.to_csv(MARTS / f"{name}.csv", index=False)

    daily["dt"] = pd.to_datetime(daily.dt)
    law = pd.Timestamp("2026-04-01")

    # 1) аудитория
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(daily.dt, daily.mau, color=COLORS["main"], label="MAU (30 дней)")
    ax.plot(daily.dt, daily.wau, color=COLORS["green"], label="WAU (7 дней)")
    ax.plot(daily.dt, daily.dau.rolling(7).mean(), color=COLORS["accent"], label="DAU (ср. за 7 дней)")
    ax.set(title="Аудитория маркетплейса", ylabel="пользователи")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "audience.png"); plt.close(fig)

    # 2) доля BNPL в GMV по неделям + дата закона
    w = daily.set_index("dt")[["gmv", "bnpl_gmv"]].resample("W-MON").sum()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(w.index, w.bnpl_gmv / w.gmv * 100, color=COLORS["main"], lw=2)
    ax.axvline(law, color=COLORS["accent"], ls="--")
    ax.text(law, ax.get_ylim()[1] * 0.98, "  1 апреля: рассрочка > 50 000 ₽ отключена", color=COLORS["accent"], va="top")
    ax.set(title="Доля оплаты частями в GMV, % (по неделям)", ylabel="%")
    fig.tight_layout(); fig.savefig(FIG / "bnpl_share.png"); plt.close(fig)

    # 3) воронка (средняя за период)
    tot = funnel[["checkout", "offer_shown", "selected", "approved"]].sum()
    labels = ["Начали чекаут", "Показали «частями»", "Выбрали «частями»", "Одобрил скоринг"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(labels[::-1], tot.values[::-1], color=COLORS["main"])
    for i, v in enumerate(tot.values[::-1]):
        ax.text(v, i, f"  {v:,.0f} ({v / tot.iloc[0]:.1%})", va="center")
    ax.set(title="Воронка BNPL, янв–сен 2026", xlim=(0, tot.iloc[0] * 1.35))
    ax.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "funnel.png"); plt.close(fig)

    # 4) когорты
    piv = cohorts.pivot(index="cohort", columns="month_offset", values="retention")
    piv.index = pd.to_datetime(piv.index).strftime("%Y-%m")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    im = ax.imshow(piv.values * 100, cmap="Blues", aspect="auto", vmin=0, vmax=40)
    for (i, j), v in np.ndenumerate(piv.values):
        if not np.isnan(v) and j > 0:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.25 else "black")
    ax.set_xticks(range(piv.shape[1])); ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set(title="Retention BNPL-клиентов по когортам (повторная покупка частями)", xlabel="месяц после первой покупки")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout(); fig.savefig(FIG / "cohorts.png"); plt.close(fig)

    # 5) юнит-экономика на заказ по винтажам
    e = econ.copy()
    per = lambda c: e[c] / e.bnpl_orders
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ru_months = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
    x = [ru_months[m - 1] for m in pd.to_datetime(e.vintage).dt.month]
    ax.bar(x, per("revenue_merchant_fee"), color=COLORS["green"], label="комиссия мерчанта")
    bottom = np.zeros(len(e))
    for c, col, lab in [("cost_funding", "#F2CF5B", "фондирование"), ("cost_credit_losses", COLORS["accent"], "кредитные потери 90+"),
                        ("cost_processing", COLORS["grey"], "процессинг"), ("cost_scoring", "#B279A2", "скоринг")]:
        ax.bar(x, -per(c), bottom=-bottom, color=col, label=lab)
        bottom += per(c)
    ax.plot(x, per("contribution"), "ko-", label="маржа на заказ")
    ax.axhline(0, color="k", lw=0.8)
    # ловушка: у молодых винтажей потери 90+ ещё не наступили, маржа завышена
    first_young = int((~e.matured).idxmax()) if (~e.matured).any() else None
    if first_young is not None:
        ax.axvspan(first_young - 0.5, len(e) - 0.5, color="k", alpha=0.06)
        ax.text(len(e) - 0.6, ax.get_ylim()[0] * 0.55, "займы ещё не созрели:\nпотери 90+ впереди", fontsize=7,
                ha="right", va="center", color=COLORS["grey"])
    ax.set(title="Юнит-экономика на один BNPL-заказ, ₽ (по месяцу выдачи)", ylabel="₽ на заказ")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "unit_economics.png"); plt.close(fig)

    m = econ[econ.matured]
    summary = dict(
        # первые 30 дней MAU ещё «набирается», поэтому все три показателя считаем без них
        avg_dau=int(daily.dau.iloc[30:].mean()), avg_mau=int(daily.mau.iloc[30:].mean()),
        stickiness=round(float((daily.dau / daily.mau).iloc[30:].mean()), 3),
        funnel=dict(cr_offer=round(tot.offer_shown / tot.checkout, 3), cr_select=round(tot.selected / tot.offer_shown, 3),
                    approval=round(tot.approved / tot.selected, 3), checkout_to_bnpl=round(tot.approved / tot.checkout, 3)),
        retention_m1=round(float(piv[1].mean()), 3), retention_m3=round(float(piv[3].mean()), 3),
        matured_vintages=dict(
            vintages=list(pd.to_datetime(m.vintage).dt.strftime("%Y-%m")),
            loss_rate=round(float(m.cost_credit_losses.sum() / m.gmv.sum()), 4),
            margin_per_order=round(float(m.contribution.sum() / m.bnpl_orders.sum()), 1),
            margin_pct_gmv=round(float(m.contribution.sum() / m.gmv.sum()), 4),
            aov=round(float(m.gmv.sum() / m.bnpl_orders.sum())),
        ),
    )
    (ROOT / "reports" / "marts_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(econ.to_string())
    return summary


if __name__ == "__main__":
    main()
