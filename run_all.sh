#!/usr/bin/env bash
set -euxo pipefail
# 環境: python3, venv推奨
python -V
# 実験スクリプトを順次実行（無ければスキップ）
[ -f src/experiments/ei_balance_test.py ] && python src/experiments/ei_balance_test.py
[ -f src/experiments/latency_tuning.py ] && python src/experiments/latency_tuning.py
[ -f src/experiments/rsnn_eval.py ] && python src/experiments/rsnn_eval.py
echo "All experiments requested have been started (or skipped if missing)."
