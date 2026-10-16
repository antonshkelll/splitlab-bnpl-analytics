from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, SQL, MARTS, FIG = ROOT / "data", ROOT / "sql", ROOT / "marts", ROOT / "reports" / "figures"
for d in (MARTS, FIG):
    d.mkdir(parents=True, exist_ok=True)

AS_OF = "2026-10-15"
LAW_DATE = "2026-04-01"

# Допущения юнит-экономики (обсуждаемые, меняются в одном месте)
ECON = dict(
    merchant_fee=0.06,       # комиссия мерчанта с GMV рассрочки
    funding_rate=0.18,       # стоимость фондирования, годовых
    processing_rate=0.012,   # эквайринг и процессинг платежей
    scoring_cost=8.0,        # ₽ за одну заявку в скоринг (бюро кредитных историй и т.п.)
)

COLORS = dict(main="#4C78A8", accent="#E45756", grey="#9aa0a6", green="#54A24B")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in ("users", "events", "orders", "installments", "ab_assignments"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{DATA / t}.parquet')")
    return con


def run_sql(con, name: str, params: dict | None = None):
    return con.execute((SQL / name).read_text(encoding="utf-8"), params or {}).df()
