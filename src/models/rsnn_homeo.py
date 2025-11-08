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
