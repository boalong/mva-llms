import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score

class AnomalyDetector:
    """
    Anomaly detection using Isolation Forest, Local Outlier Factor, and a Statistical Reference Model (SRM) 
    based on Mahalanobis distance.
    """
    def __init__(self):
        """
        Initializes the anomaly detector without pre-trained models.
        """
        self.iforest = None
        self.lof = None
        self.srm_mean = None
        self.srm_cov_inv = None

    def train_detectors(self, normal_trajectories: np.ndarray) -> None:
        """
        Trains anomaly detection models using a given dataset of normal trajectories.
        
        Parameters:
        normal_trajectories (np.ndarray): A 2D array where each row is a feature vector representing a trajectory.
        """
        normal_trajectories = np.array([t for t in normal_trajectories if t is not None])
        
        if len(normal_trajectories) == 0:
            raise ValueError("No valid normal trajectories provided for training.")

        self.iforest = IsolationForest().fit(normal_trajectories)
        self.lof = LocalOutlierFactor(novelty=True).fit(normal_trajectories)

        self.srm_mean = np.mean(normal_trajectories, axis=0)
        cov = np.cov(normal_trajectories, rowvar=False)
        self.srm_cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))  # Regularized inverse

    def srm_distance(self, x: np.ndarray) -> float:
        """
        Computes the Mahalanobis distance of a given sample from the normal distribution.
        
        Parameters:
        x (np.ndarray): A feature vector representing a trajectory.
        
        Returns:
        float: The Mahalanobis distance score.
        """
        if self.srm_mean is None or self.srm_cov_inv is None:
            raise ValueError("SRM model has not been trained.")
        
        return mahalanobis(x, self.srm_mean, self.srm_cov_inv)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluates the trained anomaly detection models using AUROC scores.
        
        Parameters:
        X (np.ndarray): A 2D array of feature vectors representing test trajectories.
        y (np.ndarray): A 1D array of binary labels (0 for normal, 1 for anomalous).
        
        Returns:
        dict: A dictionary containing AUROC scores for each detection method.
        """
        if self.iforest is None or self.lof is None:
            raise ValueError("Models must be trained before evaluation.")

        return {
            'iforest_auroc': roc_auc_score(y, self.iforest.decision_function(X)),
            'lof_auroc': roc_auc_score(y, self.lof.decision_function(X)),
            'srm_auroc': roc_auc_score(y, -np.array([self.srm_distance(x) for x in X]))
        }
