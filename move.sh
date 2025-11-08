cd /Users/Shared/Program/python310/rsnn   # ← ご自身のリポジトリルートに移動

# 1) 必要ディレクトリ作成
mkdir -p src/models src/experiments src/utils results/summaries results/logs results/plots docs

# 2) 既存ファイルを整理（存在するものだけを移動）
# もしファイル名が違う場合は適宜書き換えてください。
mv -n prototype_rsnn.py src/models/ 2>/dev/null || true
mv -n rsnn_experiments_fixed.py src/experiments/ 2>/dev/null || true
mv -n rsnn_finetune_results.json results/summaries/ 2>/dev/null || true
mv -n rsnn_finetune_small_results.json results/summaries/ 2>/dev/null || true
mv -n rsnn_final_stabilization_remaining.json results/summaries/ 2>/dev/null || true
mv -n rsnn_latency_burst_comparison.json results/summaries/ 2>/dev/null || true
mv -n rsnn_ei_grid_fast.json results/summaries/ 2>/dev/null || true
mv -n rsnn_pipeline_continuation_summary.json results/summaries/ 2>/dev/null || true
mv -n rsnn_remaining_batches_summary.json results/summaries/ 2>/dev/null || true
mv -n rsnn_latency_ei_refine_results.json results/summaries/ 2>/dev/null || true
mv -n rsnn_final_verified_summary.json results/summaries/ 2>/dev/null || true
mv -n rsnn_experiments_all*.zip results/ 2>/dev/null || true
mv -n README_rsnn_experiments.md docs/ 2>/dev/null || true

# 3) requirements.txt
cat > requirements.txt <<'PYREQ'
numpy<2.0
scikit-learn
matplotlib
pandas
PYREQ

# 4) run_all.sh - 実験を順に動かす（雛形）
cat > run_all.sh <<'SH'
#!/usr/bin/env bash
set -euxo pipefail
# 環境: python3, venv推奨
python -V
# 実験スクリプトを順次実行（無ければスキップ）
[ -f src/experiments/ei_balance_test.py ] && python src/experiments/ei_balance_test.py
[ -f src/experiments/latency_tuning.py ] && python src/experiments/latency_tuning.py
[ -f src/experiments/rsnn_eval.py ] && python src/experiments/rsnn_eval.py
echo "All experiments requested have been started (or skipped if missing)."
SH
chmod +x run_all.sh

# 5) .gitignore（簡易）
cat > .gitignore <<'GIT'
__pycache__/
*.pyc
*.pkl
*.npy
*.npz
*.zip
results/
venv/
.env
GIT

# 6) 実験スクリプト雛形（models/experiments/utils）
# model（簡易）雛形
cat > src/models/rsnn_homeo.py <<'PYMODEL'
"""
src/models/rsnn_homeo.py
RSNN with STDP + homeostasis - minimal reusable class
"""
import numpy as np
class RSNNHomeo:
    def __init__(self, n_input, n_hidden, seed=0, **kwargs):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.rng = np.random.default_rng(seed)
        # TODO: move hyperparams to kwargs
        self.W = self.rng.normal(0.5, 0.1, size=(n_hidden, n_input)).clip(0.0)
    def step(self, inp):
        # placeholder for single time-step update
        raise NotImplementedError
PYMODEL

# experiment: EI balance test (script)
cat > src/experiments/ei_balance_test.py <<'PYEXP'
"""
src/experiments/ei_balance_test.py
簡易実験ランナー（E/Iバランス探索） - 出力は results/summaries/*.json に保存
"""
import json, os
from src.models.rsnn_homeo import RSNNHomeo
def main():
    os.makedirs("results/summaries", exist_ok=True)
    # TODO: call real experiments here
    summary = {"note": "placeholder - implement grid search and save results"}
    with open("results/summaries/ei_balance_placeholder.json","w") as f:
        json.dump(summary, f, indent=2)
if __name__ == "__main__":
    main()
PYEXP

# experiment: latency tuning (script)
cat > src/experiments/latency_tuning.py <<'PYLAT'
"""
src/experiments/latency_tuning.py
Latency & burst 符号化の実験雛形
"""
import json, os
def main():
    os.makedirs("results/summaries", exist_ok=True)
    summary = {"note": "latency tuning placeholder"}
    with open("results/summaries/latency_placeholder.json","w") as f:
        json.dump(summary, f, indent=2)
if __name__ == "__main__":
    main()
PYLAT

# utils: spike encoding
cat > src/utils/spike_encoding.py <<'PYENC'
"""
src/utils/spike_encoding.py
Poisson / latency / burst encoding helpers
"""
import numpy as np
def poisson_encoding(rate_vector, dt, T, rng):
    p = np.clip(rate_vector * dt, 0.0, 1.0)
    return rng.random((T, rate_vector.size)) < p
def latency_burst_encoding(rate_vector, T, rng, burst_prob=0.6, burst_len=2):
    mins = rate_vector.min(); maxs = rate_vector.max()
    if maxs == mins:
        times = np.zeros(rate_vector.size, dtype=int)
    else:
        norm = (rate_vector - mins) / (maxs - mins)
        times = ((1.0 - norm) * (T-5)).astype(int)
    spikes = np.zeros((T, rate_vector.size), dtype=bool)
    for i,t in enumerate(times):
        spikes[t, i] = True
        if rng.random() < burst_prob:
            for b in range(1, burst_len+1):
                if t+b < T:
                    spikes[t+b, i] = True
    return spikes
PYENC

# 7) docs/README を整形コピー（既にある場合は上書きしない）
if [ ! -f docs/README_rsnn_experiments.md ]; then
  cp docs/README_rsnn_experiments.md docs/README_rsnn_experiments.md 2>/dev/null || true
fi

# 8) Git add & commit (optional)
git add -A
git commit -m "Project reorg: add src/, experiments, utils, results directories and basic scripts" || true

echo "Setup complete. Please inspect src/, results/, docs/ and adjust any mv lines if some filenames differ."
