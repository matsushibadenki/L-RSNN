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
