from data_loader import ToyPDFLoader, ExperimentalPDFLoader
from semi_nmf_algo import Decomposition
import matplotlib.pyplot as plt
import numpy as np
import typing

experiment = ExperimentalPDFLoader(
    directory="/mnt/c/Users/User/Downloads/Matt/samples", sort="name",
    r=(0, 10)
)
X, r, meta = experiment.load()

elements_loader = ToyPDFLoader(
    q=(1, 20), r=(r[0], 10), rstep=r[1]-r[0], formulas=["CsBr", "Cs4PbBr6", "CsPbBr3"], qdamp=.032
)
_, g = elements_loader.load()
k = 3
algo = Decomposition(k=k, gt=X, F=g, method="NNLS")
error_init = algo.update_rule(add_floor=True, norm=False, fix_F=True, n_iter=1)
error = algo.update_rule(add_floor=True, norm=False, fix_F=False, n_iter=999)
algo._normalize_plus_floor(floor=False, normalize=True)
print(error_init)
print(error)
print(algo.G0)
print(algo.F.shape)
print(algo.F)
algo.plotting(n_cols=k, r=r, r_lim=(0, 10), g=g)
