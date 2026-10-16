#!/usr/bin/env bash
# Полный прогон: данные → витрины и графики → оценка 283-ФЗ → A/B-тест
set -euo pipefail
cd "$(dirname "$0")"
python src/generate_data.py
python src/marts.py
python src/regulation_impact.py
python src/ab_test.py
echo "Готово: графики в reports/figures, витрины в marts/"
