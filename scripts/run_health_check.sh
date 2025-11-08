#!/usr/bin/env bash
# scripts/run_health_check.sh
# Runs the Python-based health check and visualization.

set -euo pipefail

# プロジェクトルート (このスクリプトの親ディレクトリ)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "===== RSNN (DI+LC) Health Check ====="
echo "Project root: $ROOT"
echo "Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# 1) Pythonの確認
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found in PATH" >&2
  exit 2
fi
PY_EXE=$(command -v python)
echo "Python: $($PY_EXE -V 2>&1)"

# 2) 依存関係の確認 (pip)
REQ=requirements.txt
if [ -f "$REQ" ]; then
  echo "Checking $REQ..."
  # pip freeze の出力を requiremets.txt と比較する簡易チェック
  # (注: これは厳密なバージョン比較を行わないため、目安です)
  if ! $PY_EXE -m pip freeze | grep -q -f "$REQ" >/dev/null 2>&1; then
      echo "WARNING: Not all packages from $REQ seem to be installed."
      echo "Please run: $PY_EXE -m pip install -r $REQ"
  else
      echo "Required packages seem installed (basic check)."
  fi
else
  echo "WARNING: $REQ not found. Skipping package check."
fi

# 3) Pythonヘルスチェックの実行 (JSON生成)
TOOLS_SCRIPT=tools/health_check.py
if [ ! -f "$TOOLS_SCRIPT" ]; then
    echo "ERROR: $TOOLS_SCRIPT not found." >&2
    exit 3
fi

echo -e "\n--- Running $TOOLS_SCRIPT ---"
# 修正: `!$PY_EXE` を `! $PY_EXE` に変更
if ! $PY_EXE "$TOOLS_SCRIPT"; then
    echo "ERROR: $TOOLS_SCRIPT failed." >&2
    exit 4
fi
echo "--- $TOOLS_SCRIPT finished ---"


# 4) 可視化スクリプトの実行 (PNG生成)
VIZ_SCRIPT=scripts/visualize_full_health.py
if [ ! -f "$VIZ_SCRIPT" ]; then
    echo "ERROR: $VIZ_SCRIPT not found." >&2
    exit 5
fi

echo -e "\n--- Running $VIZ_SCRIPT ---"
# 修正: `!$PY_EXE` を `! $PY_EXE` に変更
if ! $PY_EXE "$VIZ_SCRIPT"; then
    echo "ERROR: $VIZ_SCRIPT failed." >&2
    exit 6
fi
echo "--- $VIZ_SCRIPT finished ---"

echo -e "\nHealth check and visualization complete."
echo "Report JSON: outputs/health_report.json"
echo "Report PNG:  outputs/full_health_check.png"