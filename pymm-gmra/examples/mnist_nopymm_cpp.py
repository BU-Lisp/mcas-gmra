# SYSTEM IMPORTS
from tensorflow.keras.datasets import mnist
from typing import Set
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import os
import sys
import torch as pt

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from gmra_trees import CoverTree


def main() -> None:
    (X_train, _), (X_test, _) = mnist.load_data()
    X: np.ndarray = np.vstack([X_train, X_test])
    X = pt.from_numpy(X.reshape(X.shape[0], -1).astype(np.float32))

    """
    max_l2_norm: float = 0
    for i in tqdm(range(X.shape[0]), desc="computing max scale"):
        l2_dist: float = np.max(((X[i] - X[i+1:,:])**2).sum(axis=1)**(1/2))
        if max_l2_norm < l2_dist:
            l2_dist = max_l2_norm
    max_scale: int = np.ceil(np.log2(max_l2_norm))
    print("max_scale: ", max_scale)
    """

    max_scale = 13

    covertree = CoverTree(max_scale=max_scale)
    print(covertree.num_nodes, X.shape[0])
    print(covertree.min_scale, covertree.max_scale)

    for pt_idx in tqdm(list(range(X.shape[0]))[:-1], desc="building covertree"):
        covertree.insert_pt(pt_idx, X)
        # print()

    print(covertree.num_nodes, X.shape[0])
    print(covertree.min_scale, covertree.max_scale)

    """
    celltree = DyadicCellTree().from_covertree(covertree)
    celltree.check_tree()
    print("num cells", celltree.num_nodes)
    """


if __name__ == "__main__":
    main()

