"""Metrics for classification with soft (probabilistic) labels"""

import logging

import numpy as np

from . import make_scorer

logger = logging.getLogger(__name__)

EPS = 1e-10  # clipping threshold to prevent NaN


def _soft_log_loss(true_probs, predicted_probs):
    """Both args must be 2D pandas/numpy arrays"""
    true_probs = np.array(true_probs)
    predicted_probs = np.array(predicted_probs)
    if len(true_probs.shape) != 2 or len(predicted_probs.shape) != 2:
        raise ValueError("both truth and prediction must be 2D numpy arrays")
    if true_probs.shape != predicted_probs.shape:
        raise ValueError("truth and prediction must be 2D numpy arrays with the same shape")

    # true_probs = np.clip(true_probs, a_min=EPS, a_max=None)
    predicted_probs = np.clip(predicted_probs, a_min=EPS, a_max=None)  # clip 0s to avoid NaN
    true_probs = true_probs / true_probs.sum(axis=1, keepdims=1)  # renormalize
    predicted_probs = predicted_probs / predicted_probs.sum(axis=1, keepdims=1)
    loss = -(np.log(predicted_probs) * true_probs).sum(axis=1).mean()
    return loss


# Score for soft-classification (with soft, probabilistic labels):
soft_log_loss = make_scorer("soft_log_loss", _soft_log_loss, greater_is_better=False, needs_proba=True)
