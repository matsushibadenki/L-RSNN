# ./src/rsnn/core/encoding.py
# タイトル: スパイクエンコーディングモジュール
# 機能説明: レートベースの入力をスパイク列に変換する関数（Poisson符号化、Latency+Burst符号化）を提供します。
from __future__ import annotations
import numpy as np

def poisson_encoding(rate_vector: np.ndarray, dt: float, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    シンプルなPoissonスパイクジェネレータ（タイムステップ毎）。
    
    Args:
        rate_vector (np.ndarray): 入力レート (n_input,)
        dt (float): タイムステップ（秒）
        T (int): 総タイムステップ数
        rng (np.random.Generator): 乱数生成器
    
    Returns:
        np.ndarray: スパイク行列 (T, n_input)
    """
    p = np.clip(rate_vector * dt, 0.0, 1.0)
    return rng.random((T, rate_vector.size)) < p

def latency_burst_encoding(rate_vector: np.ndarray, T: int, rng: np.random.Generator,
                           burst_prob: float = 0.6, burst_len: int = 2) -> np.ndarray:
    """
    Latency + Burst 符号化。
    高レートほど早いスパイクを生成し、オプションでバースト（連続スパイク）を発生させます。
    
    Args:
        rate_vector (np.ndarray): 入力レート (n_input,)
        T (int): 総タイムステップ数
        rng (np.random.Generator): 乱数生成器
        burst_prob (float): バースト発生確率
        burst_len (int): バースト長（初期スパイクを除く追加スパイク数）
    
    Returns:
        np.ndarray: スパイク行列 (T, n_input)
    """
    mins = rate_vector.min()
    maxs = rate_vector.max()
    
    if maxs == mins:
        # レートが全て同じ場合、(T-5)ステップ目に発火
        times = np.full(rate_vector.size, T - 5, dtype=int)
    else:
        # 正規化 (0.0 - 1.0)
        norm = (rate_vector - mins) / (maxs - mins)
        # 高レート(norm=1) -> 0, 低レート(norm=0) -> T-5
        times = ((1.0 - norm) * (T - 5)).astype(int)
        
    spikes = np.zeros((T, rate_vector.size), dtype=bool)
    
    for i, t in enumerate(times):
        if t < 0: t = 0
        if t >= T: t = T - 1
            
        spikes[t, i] = True
        
        # バーストの適用
        if rng.random() < burst_prob:
            for b in range(1, burst_len + 1):
                if t + b < T:
                    spikes[t + b, i] = True
                    
    return spikes