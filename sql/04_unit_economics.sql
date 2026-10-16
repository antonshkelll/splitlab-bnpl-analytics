-- Юнит-экономика BNPL по месяцу выдачи (винтажу).
-- Доход: комиссия мерчанта. Расходы: фондирование, кредитные потери, процессинг, скоринг.
-- Параметры ($merchant_fee, $funding_rate, ...) — допущения, передаются из Python.

WITH inst AS (
    SELECT i.*, o.ts::DATE AS order_date, date_trunc('month', o.ts)::DATE AS vintage
    FROM installments i JOIN orders o USING (order_id)
),
per_order AS (
    SELECT order_id, vintage, order_date,
           sum(amount)                                                      AS amount,
           -- фондирование: каждый платёж «висит» на балансе от покупки до оплаты (или до среза)
           sum(amount * date_diff('day', order_date, coalesce(paid_date, $as_of::DATE)) / 365.0)
                                                                            AS rub_years_outstanding,
           -- потеря = неоплаченные платежи с просрочкой > 90 дней на дату среза
           sum(amount) FILTER (WHERE paid_date IS NULL AND due_date + INTERVAL 90 DAY <= $as_of::DATE)
                                                                            AS loss_90
    FROM inst GROUP BY 1, 2, 3
),
apps AS (         -- заявки в скоринг (включая отказы) — каждая стоит денег
    SELECT date_trunc('month', ts)::DATE AS vintage, count(*) AS applications
    FROM events WHERE event = 'bnpl_selected' GROUP BY 1
)
SELECT p.vintage,
       count(*)                                                        AS bnpl_orders,
       round(sum(p.amount))                                            AS gmv,
       round(sum(p.amount) * $merchant_fee)                            AS revenue_merchant_fee,
       round(sum(p.rub_years_outstanding) * $funding_rate)             AS cost_funding,
       round(coalesce(sum(p.loss_90), 0))                              AS cost_credit_losses,
       round(sum(p.amount) * $processing_rate)                         AS cost_processing,
       round(a.applications * $scoring_cost)                           AS cost_scoring,
       round(sum(p.amount) * $merchant_fee
             - sum(p.rub_years_outstanding) * $funding_rate
             - coalesce(sum(p.loss_90), 0)
             - sum(p.amount) * $processing_rate
             - a.applications * $scoring_cost)                         AS contribution,
       round(coalesce(sum(p.loss_90), 0) / sum(p.amount), 4)           AS loss_rate,
       -- винтаж «созрел», если последний платёж + 90 дней уже наступил
       max(p.order_date) + INTERVAL 42 DAY + INTERVAL 90 DAY <= $as_of::DATE AS matured
FROM per_order p JOIN apps a USING (vintage)
GROUP BY p.vintage, a.applications
ORDER BY p.vintage;
