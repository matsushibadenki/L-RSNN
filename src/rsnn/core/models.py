# ./src/rsnn/core/models.py
# タイトル: RSNNモデル実装
# 機能説明: RSNN_Homeo (STDP + 恒常性) および RSNN_EI (E/I分離 + k-Winners) の具体的な実装を提供します。
from __future__ import annotations
import numpy as np
from typing import Callable
from .base_rsnn import BaseRSNN
from .learning_rules import STDP, Homeostasis

class RSNN_Homeo(BaseRSNN):
    """STDPと恒常性（適応的閾値）を持つRSNNモデル"""
    
    def __init__(self, n_input: int, n_hidden: int, dt: float, tau_m: float, 
                 v_th: float, v_reset: float, rec_delay: int, rng_seed: int,
                 stdp_rule: STDP, homeostasis_rule: Homeostasis):
        
        super().__init__(n_input, n_hidden, dt, tau_m, v_th, v_reset, rec_delay, rng_seed)
        
        self.stdp = stdp_rule
        self.homeo = homeostasis_rule
        
        # 状態変数
        self.adaptive_theta = np.ones(self.n_hidden) * self.v_th_base

    def run_sample(self, rates: np.ndarray, T: int,
                   encoding_fn: Callable[..., np.ndarray],
                   encoding_params: dict,
                   train_stdp: bool = True) -> np.ndarray:
        
        # 状態の初期化
        V = np.zeros(self.n_hidden)
        # スパイクバッファ (遅延配信用)
        spikes_buffer = np.zeros((self.rec_delay + 2, self.n_hidden))
        hidden_record = np.zeros((T, self.n_hidden))
        
        # 入力スパイクの生成
        input_matrix = encoding_fn(rates, T=T, rng=self.rng, **encoding_params)
        
        # STDPトレースの初期化
        pre_trace = np.zeros(self.n_input)
        post_trace = np.zeros(self.n_hidden)
        pre_trace_rec = np.zeros(self.n_hidden)
        post_trace_rec = np.zeros(self.n_hidden)
        
        firing_count = np.zeros(self.n_hidden)

        for t in range(T):
            inp = input_matrix[t].astype(float)
            
            # リカレント入力（遅延を考慮）
            rec_spikes = spikes_buffer[-(self.rec_delay + 1)]
            
            I_in = self.W @ inp
            I_rec = self.U @ rec_spikes
            
            # 電圧更新 (LIF)
            V = V * self.decay_m + (I_in + I_rec) * self.scale_m
            
            # スパイク判定 (適応的閾値を使用)
            spk = (V >= self.adaptive_theta).astype(float)
            
            # リセット
            V = np.where(spk > 0, self.v_reset, V)
            
            # バッファ更新
            spikes_buffer = np.roll(spikes_buffer, -1, axis=0)
            spikes_buffer[-1] = spk
            hidden_record[t] = spk
            firing_count += spk
            
            if train_stdp:
                # 入力 -> 隠れ層 STDP
                pre_trace, post_trace, self.W = self.stdp(
                    inp, spk, pre_trace, post_trace, self.W
                )
                # 隠れ層 -> 隠れ層 STDP (1ステップ前のスパイクを使用)
                pre_trace_rec, post_trace_rec, self.U = self.stdp(
                    spikes_buffer[-2], spk, pre_trace_rec, post_trace_rec, self.U
                )
            
            # 恒常性（閾値）の更新
            self.adaptive_theta = self.homeo(self.adaptive_theta, firing_count, t)
            
        return hidden_record


class RSNN_EI(RSNN_Homeo):
    """E/I分離とk-Winners-Take-Allを持つRSNNモデル"""
    
    def __init__(self, n_input: int, n_hidden: int, dt: float, tau_m: float, 
                 v_th: float, v_reset: float, rec_delay: int, rng_seed: int,
                 stdp_rule: STDP, homeostasis_rule: Homeostasis,
                 excitatory_ratio: float = 0.8, inh_strength: float = 1.0,
                 k_winners: int = 8):
        
        super().__init__(n_input, n_hidden, dt, tau_m, v_th, v_reset, rec_delay, rng_seed,
                         stdp_rule, homeostasis_rule)
        
        self.k_winners = k_winners
        
        # E/Iニューロンのインデックス
        n_exc = int(n_hidden * excitatory_ratio)
        self.exc_idx = np.arange(n_exc)
        self.inh_idx = np.arange(n_exc, n_hidden)
        
        # リカレント重み U を E/I に基づき再構築
        # E (列: 0..n_exc-1) は正、 I (列: n_exc..) は負
        pos = np.abs(self.rng.normal(0.15, 0.04, size=(n_hidden, n_hidden)))
        U_ei = np.zeros_like(pos)
        U_ei[:, :n_exc] = pos[:, :n_exc]
        U_ei[:, n_exc:] = -np.abs(pos[:, n_exc:]) * inh_strength
        np.fill_diagonal(U_ei, 0.0)
        self.U = U_ei

    def run_sample(self, rates: np.ndarray, T: int,
                   encoding_fn: Callable[..., np.ndarray],
                   encoding_params: dict,
                   train_stdp: bool = True) -> np.ndarray:
        
        V = np.zeros(self.n_hidden)
        spikes_buffer = np.zeros((self.rec_delay + 2, self.n_hidden))
        hidden_record = np.zeros((T, self.n_hidden))
        
        input_matrix = encoding_fn(rates, T=T, rng=self.rng, **encoding_params)
        
        pre_trace = np.zeros(self.n_input); post_trace = np.zeros(self.n_hidden)
        pre_trace_rec = np.zeros(self.n_hidden); post_trace_rec = np.zeros(self.n_hidden)
        firing_count = np.zeros(self.n_hidden)
        
        n_exc = len(self.exc_idx)

        for t in range(T):
            inp = input_matrix[t].astype(float)
            rec_spikes = spikes_buffer[-(self.rec_delay + 1)]
            
            I_in = self.W @ inp
            I_rec = self.U @ rec_spikes
            
            V = V * self.decay_m + (I_in + I_rec) * self.scale_m
            
            # --- k-Winners-Take-All (Excニューロンのみ) ---
            spk = np.zeros(self.n_hidden)
            if n_exc > 0:
                exc_V = V[self.exc_idx]
                k = min(self.k_winners, n_exc)
                # 電圧が上位k個のニューロンを選択
                winners_local_idx = np.argsort(exc_V)[-k:]
                winners_global_idx = self.exc_idx[winners_local_idx]
                spk[winners_global_idx] = 1.0
            
            # --- 抑制性ニューロンの発火 (通常のスレッショルディング) ---
            if len(self.inh_idx) > 0:
                inh_spk = (V[self.inh_idx] >= self.adaptive_theta[self.inh_idx]).astype(float)
                spk[self.inh_idx] = inh_spk
                
                # 抑制性スパイクによる興奮性ニューロンへの即時抑制効果 (オプション)
                if inh_spk.sum() > 0:
                    inh_effect = (self.U @ spk)
                    V[self.exc_idx] += inh_effect[self.exc_idx] * 0.5 # type: ignore[attr-defined]

            # リセット
            V = np.where(spk > 0, self.v_reset, V)
            
            spikes_buffer = np.roll(spikes_buffer, -1, axis=0)
            spikes_buffer[-1] = spk
            hidden_record[t] = spk
            firing_count += spk
            
            if train_stdp:
                pre_trace, post_trace, self.W = self.stdp(
                    inp, spk, pre_trace, post_trace, self.W
                )
                pre_trace_rec, post_trace_rec, self.U = self.stdp(
                    spikes_buffer[-2], spk, pre_trace_rec, post_trace_rec, self.U
                )
                
                # STDP後の重みの符号制約を強制
                if self.U.shape[1] > n_exc:
                    self.U[:, :n_exc] = np.maximum(self.U[:, :n_exc], 0.0) # E (列) は正
                    self.U[:, n_exc:] = np.minimum(self.U[:, n_exc:], 0.0) # I (列) は負

            # 恒常性（E/I両方）
            self.adaptive_theta = self.homeo(self.adaptive_theta, firing_count, t)
            
        return hidden_record