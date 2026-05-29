import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Union

from src.config import IFOREST_PARAMS

class AnomalyIsolationForest:
    def __init__(self, params: Dict[str, Any] = IFOREST_PARAMS):
        self.model = IsolationForest(**params)
        self.mu = 0.0
        self.sigma = 1.0
        self.threshold = 0.5  # Calibrated threshold
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Fit Isolation Forest model strictly on clean normal transactional data.
        """
        print("Training Isolation Forest baseline model...")
        self.model.fit(X)
        
        # Calculate raw scores: -decision_function so that more anomalous = higher positive value
        raw_scores = self.model.decision_function(X)
        neg_scores = -raw_scores
        
        # Robustly prune top 2% of scores to avoid scale squashing
        q = np.percentile(neg_scores, 98.0)
        clean_scores = neg_scores[neg_scores <= q]
        self.mu = float(np.mean(clean_scores))
        self.sigma = float(np.std(clean_scores))
        if self.sigma == 0.0:
            self.sigma = 1.0
            
        self.is_fitted = True
        print(f"Isolation Forest trained successfully. Z-Score parameters: mu={self.mu:.4f}, sigma={self.sigma:.4f}")

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate continuous anomaly probability score in range [0, 1] using Soft Bounding.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")
            
        raw_scores = self.model.decision_function(X)
        neg_scores = -raw_scores
        
        # Compute Soft Bounding: Sigmoid applied to Z-Score
        z_scores = (neg_scores - self.mu) / self.sigma
        scaled_scores = 1.0 / (1.0 + np.exp(-z_scores))
        
        return scaled_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns boolean flags: True (1) if anomaly, False (0) if normal transaction
        based on calibrated threshold.
        """
        scores = self.predict_score(X)
        return (scores > self.threshold).astype(int)

    def calibrate_threshold(self, X_val: np.ndarray, contamination: float = 0.01):
        """
        Calibrate the decision boundary threshold based on a target contamination percentage.
        """
        scores = self.predict_score(X_val)
        self.threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))
        print(f"Isolation Forest calibrated Soft threshold: {self.threshold:.4f} (contamination: {contamination*100}%)")
