import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import nnls


class Decomposition():
    def __init__(
        self,
        k: int,
        gt: np.ndarray[float],
        F: np.ndarray[float] | None = None,  # F=g.T
        method: str = "KMeans",
        offset: float = .2,
        n_init: int = 10,
        random_state: int = 0,
        epsilon: float = 1e-12
    ):
        self.epsilon = epsilon
        self.G0_init = np.empty(shape=(gt.shape[0], k))
        self.F = F
        if self.F is None:
            if method == "NNLS":
                raise ValueError(
                    f"\nThe F matrix has not been initialized"
                    f"as it is not passed as one of the\n"
                    f"arguments to the object instance."
                    f"Falling back to the default initialization\n"
                    f"(F is initialized as empty matrix)"
                )
            else:
                self.F = np.zeros(shape=(gt.shape[1], k))

        if self.F.shape != (gt.shape[1], k):
            raise ValueError(
                f"\nThe initialized F matrix shape {self.F.shape}"
                f"does not correspond \nto the appropriate "
                f"shape of {(gt.shape[1], k)}"
            )
        self.k = k
        self.gt = gt
        if method == "KMeans":
            km = KMeans(n_clusters=self.k, n_init=n_init,
                        random_state=random_state)
            labels = km.fit_predict(gt)
            for i in range(self.G0_init.shape[0]):  # rows
                for j in range(self.G0_init.shape[1]):  # columns
                    if labels[i] == j:
                        self.G0_init[i, j] = 1
                    else:
                        self.G0_init[i, j] = offset
        elif method == "NNLS":
            for i in range(self.gt.shape[0]):
                self.G0_init[i], _ = nnls(self.F, self.gt[i])
        else:
            U, S, Vt = np.linalg.svd(self.gt, full_matrices=False)
            self.G0_init = U[:, :self.k] * S[:self.k]
            for i in range(self.k):
                if np.sum(self.G0_init[:, i]) < 0:
                    self.G0_init[:, i] *= -1
                if np.min(self.G0_init[:, i]) < 0:
                    self.G0_init[:, i] -= np.min(self.G0_init[:, i])
        # adding epsilon to prevent division by 0 and values of normalized G0 from blowing up
        self.G0 = self.G0_init

    def pos(
        self,
        A: np.ndarray
    ) -> np.ndarray:
        return (np.abs(A) + A) / 2.0

    def neg(
        self,
        A: np.ndarray
    ) -> np.ndarray:
        return (np.abs(A) - A) / 2.0

    def _normalize_plus_floor(
        self,
        floor,
        normalize
    ):
        if floor:
            self.G0 = np.maximum(self.G0, self.epsilon)
        if normalize:
            self.G0 = self.G0 / (self.G0.sum(axis=1, keepdims=True))

    def update_rule(
        self,
        update_G0: bool = True,
        n_iter: int = 10**4,
        fix_F: bool = False,
        norm: bool = True,
        add_floor: bool = True
    ) -> np.ndarray:
        if update_G0 is True:
            self.G0 = self.G0_init.copy()
        error = np.zeros(n_iter)
        for i in range(n_iter):
            # phase matrix: allowed to be positive and negative
            if not fix_F:
                self.F = (
                    self.gt.T @ self.G0) @ np.linalg.pinv(self.G0.T @ self.G0)
            XF = self.gt @ self.F          # shape: (5, 2)
            FtF = self.F.T @ self.F       # shape: (2, 2)

            num = self.pos(XF) + self.G0 @ self.neg(FtF)
            den = self.neg(XF) + self.G0 @ self.pos(FtF) + self.epsilon
            self.G0 *= np.sqrt(num / den)

            X_hat = self.G0 @ self.F.T    # shape: (5, 10001)
            self._normalize_plus_floor(normalize=norm, floor=add_floor)
            error[i] = np.linalg.norm(self.gt - X_hat)

        return error

    def plotting(
        self,
        r: np.ndarray,
        g: np.ndarray,
        n_cols: int = 3,
        plot_Gs: bool = True,  # (5, 3)
        compare_F: bool = True,  # (n_sample (like, 8000), 3)
        indecies_F: np.ndarray | None = None,
        norm_F_plus_g: bool = True,
        fig_size: tuple = (3.25, 6.5),
        font_size: str = '12',
        font_style: str = "DejaVu Sans",
        fig_marker: str = 'o',
        graph_style: str = 'lines',
        fig_linewidth: float = 1.0,
        r_lim: tuple = (0, 10.0)
    ):
        colors = plt.cm.tab10.colors
        if indecies_F is None:
            indecies_F = np.arange(0, n_cols+1, 1)
        plt.rcParams['figure.figsize'] = fig_size
        plt.rcParams['font.size'] = font_size
        plt.rcParams['font.sans-serif'] = [font_style]
        plt.rc(graph_style, linewidth=fig_linewidth)

        if plot_Gs:
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(n_cols):
                ax.plot(self.G0[:, i], marker=fig_marker,
                        linewidth=2, label=f"Phase {i+1}", color=colors[i])
            ax.legend()
            ax.set_xlabel("Sample")
            ax.set_ylabel("Concentration")
            fig.tight_layout()
        if compare_F:
            fig, axs = plt.subplots(1, n_cols, squeeze=False, figsize=(8, 6))
            for i in range(n_cols):
                a = axs[0, i]
                a.set_xlim(r_lim)
                if norm_F_plus_g:
                    F = self.F[:, indecies_F[i]] / \
                        max(self.F[:, indecies_F[i]])
                    gi = g[:, i]/max(g[:, i])
                else:
                    F = self.F[:, indecies_F[i]]
                    gi = g[:, i]
                a.plot(r, F, label=f"F: Phase {i+1}", color=colors[i])
                a.plot(r, gi, linestyle='-', color='k', label=f"g{i+1}")
                a.set_title(f"Phase {i+1}")
                a.legend()
            fig.tight_layout()
        plt.show()
