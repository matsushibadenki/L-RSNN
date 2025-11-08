# ./src/rsnn/core/learning_rules.py
# タイトル: 学習ルールモジュール
# 機能説明: STDP（Spike-Timing-Dependent Plasticity）および恒常性（Homeostasis）のロジックを提供します。
from __future__ import annotations
import math
import numpy as np

class STDP:
    """ペアベースSTDPの適用ロジック"""
    def __init__(self, eta: float, tau_pre: float, tau_post: float, dt: float):
        self.eta = eta
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.dt = dt
        self.exp_dt_pre = math.exp(-self.dt / self.tau_pre)
        self.exp_dt_post = math.exp(-self.dt / self.tau_post)

    def __call__(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
                 pre_trace: np.ndarray, post_trace: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        STDP更新を適用し、トレースと重みを返します。
        
        Args:
            pre_spikes (np.ndarray): プリシナプスのスパイク (0/1)
            post_spikes (np.ndarray): ポストシナプスのスパイク (0/1)
            pre_trace (np.ndarray): プリシナプスのトレース
            post_trace (np.ndarray): ポストシナプスのトレース
            W (np.ndarray): 重み行列
        
        Returns:
            tuple: (更新された pre_trace, 更新された post_trace, 更新された W)
        """
        pre_trace *= self.exp_dt_pre
        post_trace *= self.exp_dt_post
        
        pre_trace += pre_spikes
        post_trace += post_spikes
        
        # ポテンシエーション (プリが先)
        dW_plus = np.outer(post_spikes, pre_trace)
        # デプレッション (ポストが先)
        dW_minus = np.outer(post_trace, pre_spikes)
        
        delta = self.eta * (dW_plus - dW_minus)
        W += delta
        
        # 重みをクリップ
        np.clip(W, 0.0, 5.0, out=W)
        
        return pre_trace, post_trace, W

class Homeostasis:
    """レートベースの恒常性（適応的閾値）の適用ロジック"""
    def __init__(self, homeo_lr: float, homeo_target: float):
        self.homeo_lr = homeo_lr
        self.homeo_target = homeo_target

    def __call__(self, adaptive_theta: np.ndarray, firing_count: np.ndarray, t: int) -> np.ndarray:
        """
        適応的閾値を更新します。
        
        Args:
            adaptive_theta (np.ndarray): 現在の適応的閾値
            firing_count (np.ndarray): 現在までのスパイク総数
            t (int): 現在のタイムステップ (1ベース)
        
        Returns:
            np.ndarray: 更新された適応的閾値
        """
        rate_now = firing_count / (t + 1.0)
        # 変化量にクリップを適用（急激な変化を防ぐ）
        delta_theta = np.clip(self.homeo_lr * (rate_now - self.homeo_target), -0.01, 0.01)
        adaptive_theta += delta_theta
        return adaptive_theta