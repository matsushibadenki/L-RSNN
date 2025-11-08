# ./src/rsnn/experiments/dataset.py
# タイトル: トイ・データセット生成
# 機能説明: 合成的なレートベースのデータセット（2クラス分類）を作成します。
from __future__ import annotations
import numpy as np
from typing import Tuple

class DatasetGenerator:
    """トイ・データセット（レートベース）を生成します。"""
    
    def __init__(self, n_input: int, pattern_size: int, 
                 base_rate: float, pat_rate: float, rng_seed: int):
        self.n_input = n_input
        self.pattern_size = pattern_size
        self.base_rate = base_rate
        self.pat_rate = pat_rate
        self.rng = np.random.default_rng(rng_seed)

    def make_toy_rates(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        合成レートデータセットを作成します。
        クラス0: 最初の'pattern_size'ニューロンが高レート
        クラス1: 最後の'pattern_size'ニューロンが高レート
        """
        rates = np.ones((n_samples, self.n_input)) * self.base_rate
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            cls = i % 2
            labels[i] = cls
            if cls == 0:
                rates[i, :self.pattern_size] = self.pat_rate
            else:
                rates[i, -self.pattern_size:] = self.pat_rate
                
        return rates, labels