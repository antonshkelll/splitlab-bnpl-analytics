"""
Синтетические данные BNPL-сервиса «оплата частями» внутри маркетплейса (аналог Сплит / Долями).
Период: 1 января — 30 сентября 2026, срез на 15 октября 2026.

Что заложено в данные (и что потом должен «найти» анализ):
  * 1 апреля 2026 вступил в силу 283-ФЗ: BNPL-сервисы под надзором ЦБ, а сведения о рассрочках
    выше 50 000 ₽ передаются в бюро кредитных историй. Сценарий: после этой даты сервис
    перестаёт предлагать рассрочку для корзин дороже 50 000 ₽.
  * 3–30 августа 2026 — A/B-тест «Сплит на карточке товара»
    (показываем «4 × 2 490 ₽» прямо на карточке, а не только на чекауте).
  * 4 платежа раз в 2 недели, первый — сразу при покупке. Часть клиентов уходит в просрочку.

Таблицы: users, events, orders, installments, ab_assignments
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START, END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-01")
AS_OF = pd.Timestamp("2026-10-15")
LAW_DATE = pd.Timestamp("2026-04-01")
CAP = 50_000
AB_START, AB_END = pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-31")

CATS = {  # категория: (доля сессий, медиана чека, разброс)
    "electronics": (0.16, 22_000, 0.95),
    "home":        (0.14, 9_000, 0.9),
    "clothes":     (0.26, 4_200, 0.6),
    "beauty":      (0.16, 1_900, 0.55),
    "kids":        (0.14, 3_300, 0.7),
    "sport":       (0.14, 6_000, 0.8),
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(n_users: int, seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- пользователи ----------
    uid = np.arange(n_users)
    signup = START - pd.Timedelta(days=200) + pd.to_timedelta(rng.integers(0, 470, n_users), "D")
    age = np.clip(rng.normal(34, 10, n_users), 18, 75).round().astype(int)
    channel = rng.choice(["organic", "ads", "referral", "partner"], n_users, p=[0.45, 0.30, 0.15, 0.10])
    city = rng.choice(["msk_spb", "million_plus", "regions"], n_users, p=[0.3, 0.3, 0.4])
    credit = np.clip(rng.normal(0, 1, n_users), -3, 3)       # латентный кредитный риск (выше — надёжнее)
    activity = rng.lognormal(np.log(0.045), 0.9, n_users)    # вероятность визита в день
    bnpl_affinity = rng.normal(0, 0.8, n_users)              # склонность платить частями
    users = pd.DataFrame(dict(user_id=uid, signup_date=signup.normalize(), age=age,
                              channel=channel, city=city))

    # A/B: детерминированная рандомизация по хешу user_id (соль = название эксперимента)
    h = pd.util.hash_array(np.char.add("split_on_pdp_", uid.astype(str)).astype(object))
    ab_group = np.where(h % 100 < 50, "control", "treatment")
    ab = pd.DataFrame(dict(user_id=uid, experiment="split_on_pdp", group=ab_group))

    # ---------- сессии ----------
    days = (END - START).days
    k = rng.poisson(np.minimum(activity, 0.6) * days)
    s_user = np.repeat(uid, k)
    s_day = rng.integers(0, days, len(s_user))
    s_ts = START + pd.to_timedelta(s_day, "D") + pd.to_timedelta(rng.uniform(7, 24, len(s_user)) * 3600, "s")
    # в выходные активность выше: часть будних сессий отбрасываем
    keep = rng.random(len(s_user)) < np.where(s_ts.dayofweek >= 5, 1.0, 0.85)
    keep &= s_ts >= signup[s_user]
    s_user, s_ts = s_user[keep], s_ts[keep]
    n_s = len(s_user)

    cat_names = list(CATS)
    cat = rng.choice(cat_names, n_s, p=[CATS[c][0] for c in cat_names])
    med = np.array([CATS[c][1] for c in cat]); sd = np.array([CATS[c][2] for c in cat])
    amount = np.round(rng.lognormal(np.log(med), sd), -1)

    in_ab = (s_ts >= AB_START) & (s_ts < AB_END)
    treat = in_ab & (ab_group[s_user] == "treatment")

    # чекаут: на карточке товара видно «4 × N ₽» → чуть чаще доходят до оформления
    p_checkout = sigmoid(-1.75 - 0.08 * np.log(amount / 3000) + 0.05 * treat)
    checkout = rng.random(n_s) < p_checkout

    # предложение BNPL на чекауте: всегда до 1 апреля; после — только если корзина ≤ 50 000 ₽
    offered = checkout & ((s_ts < LAW_DATE) | (amount <= CAP))
    p_select = sigmoid(-1.25 + 0.38 * np.log(amount / 5000) + bnpl_affinity[s_user]
                       - 0.015 * (age[s_user] - 34) + 0.15 * treat)
    selected = offered & (rng.random(n_s) < p_select)
    p_approve = sigmoid(1.6 + 1.1 * credit[s_user] - 0.25 * np.log(amount / 5000))
    approved = selected & (rng.random(n_s) < p_approve)
    declined = selected & ~approved

    # оплата: BNPL / карта / СБП; после отказа скоринга часть людей бросает корзину
    pay_card = checkout & ~approved & (rng.random(n_s) < np.where(declined, 0.55, 0.80))
    paid = approved | pay_card
    method = np.where(approved, "bnpl", np.where(rng.random(n_s) < 0.7, "card", "sbp"))

    # ---------- events ----------
    def ev(mask, name, lag_s):
        return pd.DataFrame(dict(user_id=s_user[mask], event=name,
                                 ts=s_ts[mask] + pd.to_timedelta(lag_s, "s"),
                                 category=cat[mask], amount=amount[mask]))
    events = pd.concat([
        ev(np.ones(n_s, bool), "session_start", 0),
        ev(checkout, "checkout_start", 90),
        ev(offered, "bnpl_offer_shown", 95),
        ev(selected, "bnpl_selected", 120),
        ev(approved, "bnpl_approved", 125),
        ev(declined, "bnpl_declined", 125),
        ev(paid, "order_paid", 180),
    ], ignore_index=True).sort_values("ts", kind="stable")
    events.insert(0, "event_id", np.arange(len(events)))

    # ---------- orders ----------
    orders = pd.DataFrame(dict(user_id=s_user[paid], ts=s_ts[paid] + pd.Timedelta(seconds=180),
                               category=cat[paid], amount=amount[paid], payment_method=method[paid],
                               in_ab_treatment=treat[paid]))
    orders = orders.sort_values("ts").reset_index(drop=True)
    orders.insert(0, "order_id", np.arange(len(orders)))

    # ---------- installments (4 платежа, раз в 14 дней, первый сразу) ----------
    b = orders[orders.payment_method == "bnpl"]
    nb = len(b)
    ub = b.user_id.to_numpy()
    # риск дефолта: кредитный скор + размер покупки; в тесте риск чуть выше (импульсные покупки)
    p_bad = sigmoid(-3.35 - 1.0 * credit[ub] + 0.30 * np.log(b.amount.to_numpy() / 5000)
                    + 0.12 * b.in_ab_treatment.to_numpy())
    bad = rng.random(nb) < p_bad
    first_missed = np.where(bad, rng.integers(2, 5, nb), 99)          # с какого платежа перестал платить
    late_but_paid = (~bad) & (rng.random(nb) < 0.12)                   # просрочка на 1–20 дней, но вернул
    inst = []
    for part in range(1, 5):
        due = b.ts.dt.normalize() + pd.to_timedelta((part - 1) * 14, "D")
        lag = np.where(part == 1, 0, np.where(late_but_paid & (rng.random(nb) < 0.4),
                                              rng.integers(1, 21, nb), rng.integers(-3, 1, nb)))
        paid_date = due + pd.to_timedelta(lag, "D")
        # должник может частично вернуться к оплате через 30–120 дней (коллекшн)
        recovered = rng.random(nb) < 0.25
        paid_date = paid_date.where(part < first_missed,
                                    (due + pd.to_timedelta(rng.integers(30, 120, nb), "D")).where(recovered))
        paid_date = paid_date.where(paid_date <= AS_OF)
        inst.append(pd.DataFrame(dict(order_id=b.order_id.to_numpy(), part_no=part,
                                      due_date=due.to_numpy(), amount=np.round(b.amount.to_numpy() / 4, 2),
                                      paid_date=paid_date.to_numpy())))
    installments = pd.concat(inst, ignore_index=True)
    installments = installments[installments.due_date <= AS_OF + pd.Timedelta(days=60)]

    users.to_parquet(out / "users.parquet", index=False)
    ab.to_parquet(out / "ab_assignments.parquet", index=False)
    events.to_parquet(out / "events.parquet", index=False)
    orders.drop(columns="in_ab_treatment").to_parquet(out / "orders.parquet", index=False)
    installments.to_parquet(out / "installments.parquet", index=False)
    print(f"users={len(users):,} events={len(events):,} orders={len(orders):,} "
          f"bnpl_orders={nb:,} installments={len(installments):,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    a = p.parse_args()
    main(a.users, a.seed, a.out)
