-- Ежедневные продуктовые метрики: DAU / WAU / MAU, stickiness, заказы, GMV, доля BNPL.
-- WAU и MAU считаются честно — как число уникальных пользователей в скользящем окне 7/30 дней,
-- а не суммой DAU (частая ошибка на собеседованиях).

WITH days AS (
    SELECT d::DATE AS dt
    FROM generate_series(DATE '2026-01-01', DATE '2026-09-30', INTERVAL 1 DAY) t(d)
),
active AS (       -- уникальные пары (пользователь, день активности)
    SELECT DISTINCT user_id, ts::DATE AS dt
    FROM events WHERE event = 'session_start'
),
rolling AS (
    SELECT d.dt,
           count(DISTINCT a.user_id) FILTER (WHERE a.dt = d.dt)                  AS dau,
           count(DISTINCT a.user_id) FILTER (WHERE a.dt > d.dt - INTERVAL 7 DAY)  AS wau,
           count(DISTINCT a.user_id)                                             AS mau
    FROM days d
    JOIN active a ON a.dt > d.dt - INTERVAL 30 DAY AND a.dt <= d.dt
    GROUP BY d.dt
),
ord AS (
    SELECT ts::DATE AS dt,
           count(*)                                                AS orders,
           sum(amount)                                             AS gmv,
           count(*) FILTER (WHERE payment_method = 'bnpl')         AS bnpl_orders,
           sum(amount) FILTER (WHERE payment_method = 'bnpl')      AS bnpl_gmv,
           count(DISTINCT user_id) FILTER (WHERE payment_method = 'bnpl') AS bnpl_buyers
    FROM orders GROUP BY 1
)
SELECT r.dt,
       r.dau, r.wau, r.mau,
       round(r.dau / r.mau, 4)                       AS stickiness_dau_mau,
       o.orders, o.gmv, o.bnpl_orders, o.bnpl_gmv, o.bnpl_buyers,
       round(o.bnpl_gmv / o.gmv, 4)                  AS bnpl_gmv_share,
       round(o.bnpl_gmv / o.bnpl_orders, 0)          AS bnpl_aov
FROM rolling r
LEFT JOIN ord o USING (dt)
ORDER BY r.dt;
