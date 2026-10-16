-- Воронка оплаты частями по месяцам: от начала чекаута до оплаченного BNPL-заказа.
-- Считаем на уровне попыток оформления (checkout), чтобы видеть, где именно теряем людей.

WITH f AS (
    SELECT date_trunc('month', ts)::DATE AS month,
           count(*) FILTER (WHERE event = 'checkout_start')    AS checkout,
           count(*) FILTER (WHERE event = 'bnpl_offer_shown')  AS offer_shown,
           count(*) FILTER (WHERE event = 'bnpl_selected')     AS selected,
           count(*) FILTER (WHERE event = 'bnpl_approved')     AS approved,
           count(*) FILTER (WHERE event = 'bnpl_declined')     AS declined
    FROM events
    WHERE ts < TIMESTAMP '2026-10-01'
    GROUP BY 1
)
SELECT month,
       checkout, offer_shown, selected, approved,
       round(offer_shown / checkout, 4)  AS cr_offer,       -- покрытие: кому вообще показали рассрочку
       round(selected / offer_shown, 4)  AS cr_select,      -- выбрали «частями»
       round(approved / selected, 4)     AS approval_rate,  -- одобрил скоринг
       round(approved / checkout, 4)     AS cr_checkout_to_bnpl
FROM f
ORDER BY month;
