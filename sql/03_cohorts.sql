-- Когортный retention BNPL-клиентов: когорта = месяц первой покупки частями,
-- retention_m = доля когорты, сделавшая BNPL-покупку через m месяцев.

WITH bnpl AS (
    SELECT user_id, date_trunc('month', ts)::DATE AS month
    FROM orders WHERE payment_method = 'bnpl'
    GROUP BY 1, 2
),
first AS (
    SELECT user_id, min(month) AS cohort FROM bnpl GROUP BY 1
),
activity AS (
    SELECT f.cohort,
           date_diff('month', f.cohort, b.month) AS month_offset,
           count(DISTINCT b.user_id)              AS users
    FROM bnpl b JOIN first f USING (user_id)
    GROUP BY 1, 2
)
SELECT a.cohort,
       a.month_offset,
       a.users,
       first_value(a.users) OVER (PARTITION BY a.cohort ORDER BY a.month_offset) AS cohort_size,
       round(a.users / first_value(a.users) OVER (PARTITION BY a.cohort ORDER BY a.month_offset), 4) AS retention
FROM activity a
ORDER BY a.cohort, a.month_offset;
