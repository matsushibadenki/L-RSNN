# ./src/rsnn/core/layers.py
# タイトル: SNN層モジュール
# 機能説明: SNNの基本的な層（LIFニューロン層、全結合層）のロジックを定義します。
#           (Objective 1.2 / 2.2 の基盤)
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

class BaseLayer(ABC):
    """SNN層の抽象基底クラス"""
    
    @abstractmethod
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """
        1タイムステップの順伝播を実行します。
        
        Args:
            input_data (np.ndarray): この層への入力（スパイクまたは電流）
        
        Returns:
            np.ndarray: この層の出力（電流またはスパイク）
        """
        pass

    @abstractmethod
    def reset(self):
        """層の状態（電圧など）をリセットします。"""
        pass

# --- 修正: FCLayer (全結合層) を追加 (Objective 1.2 / 2.2) ---
class FCLayer(BaseLayer):
    """
    全結合層（Fully Connected Layer）。
    重み (W) を保持し、入力スパイクとの行列積（電流）を計算します。
    この層自体は時間的な状態（電圧など）を持ちません。
    """
    
    def __init__(self, n_input: int, n_output: int, rng: np.random.Generator):
        self.n_input = n_input
        self.n_output = n_output
        self.rng = rng
        
        # 重みの初期化
        self.W = self.rng.normal(0.5, 0.1, size=(n_output, n_input)).clip(min=0.0)

    def __call__(self, input_spikes: np.ndarray) -> np.ndarray:
        """
        順伝播を実行します (I = W @ S)。
        
        Args:
            input_spikes (np.ndarray): 入力スパイクベクトル (n_input,)
        
        Returns:
            np.ndarray: 出力電流ベクトル (n_output,)
        """
        if input_spikes.shape != (self.n_input,):
             # バッチ処理（(Batch, n_input)）にも対応（オプション）
            if input_spikes.ndim == 2 and input_spikes.shape[1] == self.n_input:
                return (self.W @ input_spikes.T).T # (Batch, n_output)
            raise ValueError(f"Input shape {input_spikes.shape} mismatch. Expected ({self.n_input},)")
            
        return self.W @ input_spikes

    def reset(self):
        """状態を持たないため、何も行いません。"""
        pass

# --- LIF層 (既存) ---
class LIFLayer(BaseLayer):
    """
    Leaky Integrate-and-Fire (LIF) ニューロン層。
    ニューロンのダイナミクス（電圧更新、発火、リセット）をカプセル化します。
    """
    
    def __init__(self, 
                 shape: tuple[int, ...], 
                 dt: float, 
                 tau_m: float, 
                 v_th: float,
                 v_reset: float):
        
        self.shape = shape
        self.dt = dt
        self.tau_m = tau_m
        
        self.v_th_base: float | np.ndarray = v_th
        self.v_th: float | np.ndarray = v_th 
        
        self.v_reset = v_reset
        
        # 電圧減衰と入力スケールの係数
        self.decay_m = (1.0 - self.dt / self.tau_m)
        self.scale_m = (self.dt / self.tau_m)
        
        # 状態変数
        self.V: np.ndarray = np.zeros(shape)
        self.spikes: np.ndarray = np.zeros(shape)

    def __call__(self, I_in: np.ndarray) -> np.ndarray:
        """
        1タイムステップのLIFダイナミクスを実行します。
        
        Args:
            I_in (np.ndarray): 入力電流 (形状は self.shape と一致)
        
        Returns:
            np.ndarray: 出力スパイク (self.shape)
        """
        if I_in.shape != self.shape:
            raise ValueError(f"Input current shape {I_in.shape} must match layer shape {self.shape}")
            
        # 1. 電圧更新 (LIF)
        self.V = self.V * self.decay_m + I_in * self.scale_m
        
        # 2. スパイク判定 (v_thがfloatでもndarrayでも動作)
        self.spikes = (self.V >= self.v_th).astype(float)
        
        # 3. リセット
        self.V = np.where(self.spikes > 0, self.v_reset, self.V)
        
        return self.spikes

    def reset(self):
        """電圧とスパイクをリセットし、閾値をベース値に戻します。"""
        self.V = np.zeros(self.shape)
        self.spikes = np.zeros(self.shape)
        self.v_th = self.v_th_base
