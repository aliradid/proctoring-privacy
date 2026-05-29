"""Suspicion-Index Fusion Models.

Five fusion strategies are exposed and compared in the experiments:

    1. WeightedSumFusion – the original SI = alpha*V + beta*A + gamma*B rule.
    2. LogisticFusion    – L2-regularised logistic regression on event features.
    3. RandomForestFusion- Random forest classifier with calibrated probability.
    4. MLPFusion         - Two-layer MLP trained with early stopping.
    5. BayesianFusion    - Naive Bayes with empirical priors.

Each model implements .fit(X, y), .predict(X), .predict_proba(X) so they can be
swapped transparently. A common feature_vector(scenario_features) helper
guarantees identical inputs across fusion variants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from config import (
    FUSION_ALPHA,
    FUSION_BETA,
    FUSION_GAMMA,
    SI_FLAG_THRESHOLD,
    SI_WARN_THRESHOLD,
)

FEATURE_ORDER = [
    "max_phone_conf",
    "phone_duration_ratio",
    "max_book_conf",
    "person_count_max",
    "max_gaze_dev",
    "sustained_offscreen_ratio",
    "max_yaw_abs",
    "lip_variance_mean",
    "max_secondary_speaker_prob",
    "secondary_speaker_event_ratio",
    "max_overlap_prob",
    "max_whisper_prob",
]


def feature_vector(features: dict) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in FEATURE_ORDER], dtype=np.float32)


def _visual_score(f: dict) -> float:
    obj = max(f.get("max_phone_conf", 0.0), f.get("max_book_conf", 0.0))
    obj += 0.3 * min(1.0, f.get("person_count_max", 0.0) - 1.0) if f.get("person_count_max", 0) > 1 else 0.0
    obj = min(1.0, obj)
    gaze = min(1.0, f.get("sustained_offscreen_ratio", 0.0) + 0.5 * f.get("max_gaze_dev", 0.0) / 40.0)
    return float(0.6 * obj + 0.4 * gaze)


def _acoustic_score(f: dict) -> float:
    return float(min(1.0, max(
        f.get("max_secondary_speaker_prob", 0.0),
        0.85 * f.get("max_overlap_prob", 0.0),
        0.7 * f.get("max_whisper_prob", 0.0),
    )))


def _behaviour_score(f: dict) -> float:
    lip = min(1.0, f.get("lip_variance_mean", 0.0) / 0.005)
    yaw = min(1.0, f.get("max_yaw_abs", 0.0) / 35.0)
    return float(0.55 * yaw + 0.45 * lip)


class WeightedSumFusion:
    name = "weighted_sum"

    def __init__(self, alpha: float = FUSION_ALPHA, beta: float = FUSION_BETA, gamma: float = FUSION_GAMMA):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, X: Sequence[dict] | np.ndarray, y=None):
        return self

    def predict_proba_feat(self, features: dict) -> float:
        v = _visual_score(features)
        a = _acoustic_score(features)
        b = _behaviour_score(features)
        return float(min(1.0, self.alpha * v + self.beta * a + self.gamma * b))

    def predict_proba(self, X: Sequence[dict]) -> np.ndarray:
        scores = np.array([self.predict_proba_feat(x) for x in X])
        return np.stack([1 - scores, scores], axis=1)

    def predict(self, X: Sequence[dict]) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= SI_WARN_THRESHOLD).astype(int)


class _SkWrapper:
    name = "sk_wrapper"

    def __init__(self, model):
        self.model = model

    def fit(self, X: Sequence[dict] | np.ndarray, y: np.ndarray):
        X = np.array([feature_vector(x) if isinstance(x, dict) else x for x in X])
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: Sequence[dict] | np.ndarray) -> np.ndarray:
        X = np.array([feature_vector(x) if isinstance(x, dict) else x for x in X])
        return self.model.predict_proba(X)

    def predict(self, X: Sequence[dict] | np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_logistic_fusion():
    from sklearn.linear_model import LogisticRegression

    m = _SkWrapper(LogisticRegression(max_iter=500, C=1.0, class_weight="balanced"))
    m.name = "logistic"
    return m


def make_rf_fusion():
    from sklearn.ensemble import RandomForestClassifier

    m = _SkWrapper(RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"))
    m.name = "random_forest"
    return m


def make_mlp_fusion():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=1500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                    learning_rate_init=0.001,
                ),
            ),
        ]
    )
    m = _SkWrapper(pipe)
    m.name = "mlp"
    return m


def make_bayes_fusion():
    from sklearn.naive_bayes import GaussianNB

    m = _SkWrapper(GaussianNB())
    m.name = "naive_bayes"
    return m


def make_all_fusion_models():
    return {
        "weighted_sum": WeightedSumFusion(),
        "logistic": make_logistic_fusion(),
        "random_forest": make_rf_fusion(),
        "mlp": make_mlp_fusion(),
        "naive_bayes": make_bayes_fusion(),
    }
