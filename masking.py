import torch
import random
import numpy as np


class GilbertElliottModel:
    def __init__(self, loss_rate=None, p_range=(0.2, 0.8), q_range=(0.2, 0.8), initial_state=None):
        if loss_rate is not None:

            q_vals = np.linspace(q_range[0], q_range[1], 1000)
            p_vals = (loss_rate * q_vals) / (1 - loss_rate)
            mask = (p_vals >= p_range[0]) & (p_vals <= p_range[1])
            valid_p = p_vals[mask]
            valid_q = q_vals[mask]

            if len(valid_p) == 0:
                raise ValueError("No valid (p, q) found for given loss rate and ranges.")

            idx = np.random.randint(len(valid_p))
            self.p, self.q = valid_p[idx], valid_q[idx]

        else:
            self.p = np.random.uniform(*p_range)
            self.q = np.random.uniform(*q_range)
            loss_rate = self.p / (self.p + self.q)

        if initial_state is None:
            self.state = np.random.choice([0, 1], p=[1 - loss_rate, loss_rate])
        else:
            self.state = initial_state

    def step(self):
        if self.state == 0:
            self.state = np.random.choice([0, 1], p=[1 - self.p, self.p])
        else:
            self.state = np.random.choice([0, 1], p=[self.q, 1 - self.q])
        return self.state

    def simulate(self, spec_dim, spec_len, burn_in=10):
        trace = [self.step() for _ in range(spec_len + burn_in)]
        empirical_r = sum(trace) / len(trace)
        trace = np.array([1 - x for x in trace[burn_in:]])

        mask = np.repeat(trace[:, None], spec_dim, axis=1)
        return np.transpose(mask) #, empirical_r






