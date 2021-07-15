# SYSTEM IMPORTS
import numpy as np


# PYTHON PROJECT IMPORTS
from gmra_trees import DyadicTree


WaveletNodeType = "WaveLetNode"


class WaveletNode(object):
    def __init__(self,
                 idxs: np.ndarray,
                 X: np.ndarray,
                 tau: int) -> None:
        self.idxs: np.ndarray = idxs
        self.tau: int = tau
        self.children: List[WaveletNodeType] = list()

        # get data and mean center
        X = X[idxs]
        self.center: np.ndarray = X.mean(axis=0, keepdims=True)

        X -= self.center
        self.cov = X.dot(X.T)

        # svd on this node
        U, sigma, _ = np.linalg.svd(self.cov, full_matrices=False)
        self.U = U[:, :self.tau]
        self.sigma = sigma[:self.tau]


class WaveletTree(object):
    def __init__(self,
                 dyadic_tree: DyadicTree,
                 X: np.ndarray) -> None:
        self.

