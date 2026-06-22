"""
Semi-NMF as a predictive-coding generative model (FabricPC / JAX).

Generative model:  X_t  ≈  F @ ReLU(G_t)
  - latent node  G  (shape k): concentrations, inferred per sample; ReLU -> nonneg.
  - decoder node X  (shape n_r): observation, clamped to data; weight W (k x n_r),
    structure matrix  F = W.T  (n_r x k, mixed-sign, unconstrained).

PC infers G per sample (no encoder to overfit), the outer loop learns F.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.nodes import Linear
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import graph, TaskMap
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import InferenceSGD, run_inference


class SemiNMFDecoder(Linear):
    """Decoder: z_mu = softplus(G) @ W  (identity activation, no bias). F = W.T.
    softplus (not ReLU) keeps concentrations >= 0 while giving nonzero gradient
    everywhere, so PC inference can't get stuck in a dead-ReLU zone."""

    @staticmethod
    def forward(params, inputs, state, node_info):
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        pre = jnp.zeros((batch_size,) + out_shape)
        for edge_key, x in inputs.items():
            pre = pre + jnp.matmul(jax.nn.softplus(x), params.weights[edge_key])
        z_mu = pre  # identity activation (mixed-sign PDF)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


def build_graph(n_r, k, eta_infer=0.1, infer_steps=50):
    G = Linear(shape=(k,), activation=IdentityActivation(), name="G", use_bias=False)
    X = SemiNMFDecoder(shape=(n_r,), activation=IdentityActivation(), name="X",
                       use_bias=False)
    structure = graph(
        nodes=[G, X],
        edges=[Edge(source=G, target=X.slot("in"))],
        task_map=TaskMap(x=X),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )
    return structure, G, X


if __name__ == "__main__":
    n_r, k, batch = 50, 2, 16
    struct, G, X = build_graph(n_r, k)
    key = jax.random.PRNGKey(0)
    params = initialize_params(struct, key)
    print("param node keys:", list(params.nodes.keys()) if hasattr(params, "nodes") else type(params))
    Xdata = jnp.asarray(np.random.randn(batch, n_r).astype("float32"))
    clamps = {"X": Xdata}
    init_state = initialize_graph_state(struct, batch, key, clamps=clamps, params=params)
    final = run_inference(params, init_state, clamps, struct)
    G_inf = np.asarray(final.nodes["G"].z_latent)
    e0 = float(sum(np.sum(np.asarray(init_state.nodes[n].energy)) for n in struct.nodes))
    e1 = float(sum(np.sum(np.asarray(final.nodes[n].energy)) for n in struct.nodes))
    print(f"energy: init={e0:.3f} -> final={e1:.3f}  (should decrease)")
    print("G inferred:", G_inf.shape, "min", round(float(G_inf.min()), 3),
          "max", round(float(G_inf.max()), 3))
