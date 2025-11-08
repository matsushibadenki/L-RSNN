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
