#!/usr/bin/env bash
set -euxo pipefail
# 環境: python3, venv推奨
python -V

# ★ 修正: PYTHONPATHにプロジェクトルートを追加
# プロジェクトルート (このスクリプトがある場所)
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH set to: $PYTHONPATH" # デバッグ用にパスを表示

# 実験スクリプトを順次実行（無ければスキップ）
[ -f src/experiments/ei_balance_test.py ] && python src/experiments/ei_balance_test.py
[ -f src/experiments/latency_tuning.py ] && python src/experiments/latency_tuning.py
[ -f src/experiments/rsnn_eval.py ] && python src/experiments/rsnn_eval.py
echo "All experiments requested have been started (or skipped if missing)."
