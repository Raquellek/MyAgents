"""
utils.py — helper utilities
"""
import numpy as np


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_true = np.var(y_true)
    if var_true < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_true)


class RunningMeanStd:
    """Welford online algorithm for running mean and variance."""
    def __init__(self, epsilon: float = 1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean  = float(np.mean(x))
        batch_var   = float(np.var(x))
        batch_count = x.size
        delta       = batch_mean - self.mean
        tot         = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / tot
        m_a         = self.var * self.count
        m_b         = batch_var * batch_count
        self.var    = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.count  = tot

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        self.update(x)
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                       -clip, clip).astype(np.float32)