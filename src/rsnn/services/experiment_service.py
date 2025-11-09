# ./src/rsnn/services/experiment_service.py
# タイトル: 実験実行サービス
# 機能説明: 注入されたコンポーネント（モデル、データ、評価器）を使用して、
#           単一の実験（例：HomeoモデルでPoisson符号化）を実行する責務を持ちます。
from __future__ import annotations
import numpy as np
from typing import Callable, List, Dict, Any, TYPE_CHECKING
from ..core.base_rsnn import BaseRSNN
from ..experiments.dataset import DatasetGenerator
from ..experiments.evaluation import ReadoutEvaluator

# 修正: Factory (Provider) の型ヒントのため
if TYPE_CHECKING:
    from dependency_injector.providers import Provider

class ExperimentService:
    """
    単一の実験構成（モデル、データ、エンコーディング）を実行し、
    訓練と評価を行うサービス。
    """
    
    def __init__(self, 
                 dataset_generator: DatasetGenerator,
                 evaluator: ReadoutEvaluator):
        self.dataset_generator = dataset_generator
        self.evaluator = evaluator
        # 内部状態（データ）
        self.train_rates: np.ndarray | None = None
        self.train_labels: np.ndarray | None = None
        self.test_rates: np.ndarray | None = None
        self.test_labels: np.ndarray | None = None

    def _load_data(self, n_train: int, n_test: int):
        """データセットを（再）生成"""
        self.train_rates, self.train_labels = self.dataset_generator.make_toy_rates(n_train)
        self.test_rates, self.test_labels = self.dataset_generator.make_toy_rates(n_test)

    def run_experiment(
        self,
        model_provider: Provider[BaseRSNN], # 修正: rsnn_model -> model_provider
        encoding_fn: Callable[..., np.ndarray],
        encoding_params: dict,
        dataset_params: dict,
        sim_params: dict,
        seeds: List[int]
    ) -> List[Dict[str, Any]]:
        """
        指定されたモデルとエンコーディングで実験（複数シード）を実行します。
        
        Args:
            model_provider (Provider): DIコンテナが初期化したモデルFactory
            encoding_fn (Callable): 符号化関数
            encoding_params (dict): 符号化パラメータ
            dataset_params (dict): データセットパラメータ (n_train, n_test)
            sim_params (dict): シミュレーションパラメータ (T, epochs)
            seeds (List[int]): 実行するシード値のリスト

        Returns:
            List[Dict[str, Any]]: 各シードの結果（acc, mean_rate, mean_total_spikes）
        """
        
        # 1. データのロード
        self._load_data(dataset_params['n_train'], dataset_params['n_test'])
        
        if self.train_rates is None or self.train_labels is None or \
           self.test_rates is None or self.test_labels is None:
            raise ValueError("Data not loaded correctly.")

        T = sim_params['T']
        epochs = sim_params['epochs']
        
        results = []
        
        print(f"Running experiment (Model: {model_provider.cls.__name__}, Encoding: {encoding_fn.__name__}, Seeds: {seeds})...") # type: ignore[attr-defined]
        
        # 修正: 複数シードでループ
        for seed_val in seeds:
            
            # 2. シード毎に新しいモデルインスタンスを生成
            # Factory(rng_seed=...) で __init__ の引数を上書き
            rsnn_model = model_provider(rng_seed=seed_val)
            print(f"  Seed {seed_val}/{len(seeds)} (Model: {rsnn_model.__class__.__name__})...")
            
            # 3. 訓練 (STDP)
            for ep in range(epochs):
                print(f"    Epoch {ep+1}/{epochs}...")
                idx = rsnn_model.rng.permutation(self.train_rates.shape[0])
                for i in idx:
                    _ = rsnn_model.run_sample(
                        self.train_rates[i], T, encoding_fn, encoding_params, train_stdp=True
                    )
            
            # 4. 隠れ層の活動を収集
            print("    Collecting hidden activity...")
            H_train, _ = rsnn_model.collect_hidden_activity(
                self.train_rates, T, encoding_fn, encoding_params
            )
            H_test, test_total_spikes = rsnn_model.collect_hidden_activity(
                self.test_rates, T, encoding_fn, encoding_params
            )
            
            # 5. リードアウトの訓練と評価
            # 評価器 (Evaluator) もシード毎にリセット（Factoryで生成）
            evaluator = self.evaluator.provider() # type: ignore[attr-defined]
            evaluator.train(H_train, self.train_labels)
            acc, _ = evaluator.evaluate(H_test, self.test_labels)
            
            mean_rate = float(H_test.mean())
            mean_total_spikes = float(test_total_spikes.mean())

            results.append({
                'seed': seed_val, 
                'acc': acc, 
                'mean_rate': mean_rate,
                'mean_total_spikes': mean_total_spikes
            })
            
        return results
