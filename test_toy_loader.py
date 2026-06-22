from data_loader import ToyPDFLoader
from kinetics_init import ReactionSimulation
from semi_nmf_algo import Decomposition
import numpy as np


load_elements = ToyPDFLoader(
    formulas=["CuO", "Cu2O", "Cu"],
    q=(1, 20),
    r=(1, 10)
)
r, g = load_elements.load()

t = np.linspace(0, 50, 500)
simulate_kinematics = ReactionSimulation(g, t)
X, G = simulate_kinematics.simulation()

k = 3
reaction = Decomposition(k=k, gt=X, F=g, method="NNLS")
error = reaction.update_rule(add_floor=False, norm=False, n_iter=10)
reaction._normalize_plus_floor(floor=False, normalize=True)
reaction.plotting(r=r, g=g, n_cols=k)

print(error)
print(reaction.G0)

# formulas=["CsBr", "Cs4PbBr6", "CsPbBr3"],
