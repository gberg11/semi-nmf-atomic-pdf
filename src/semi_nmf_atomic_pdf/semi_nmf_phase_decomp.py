import numpy as np
from sklearn.cluster import KMeans

def pos(A: np.ndarray) -> np.ndarray:
    return (np.abs(A) + A) / 2.0

def neg(A: np.ndarray) -> np.ndarray:
    return (np.abs(A) - A) / 2.0

def semi_nmf_gr_phases(X, k=2, n_iter=200, offset=1e-8):
    n_features = X.shape[0]
    X = np.asarray(X, dtype=float)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    G = np.zeros((n_features, k), dtype=float)
    G[np.arange(n_features), labels] = 1.0
    G = G + offset 
    errors=[]
    for it in range(n_iter):
        #update F
        F = (X.T @ G) @ np.linalg.pinv(G.T @ G)
        XF = X @ F          
        FtF = F.T @ F     
        #update G
        num = pos(XF) + G @ neg(FtF)
        den = neg(XF) + G @ pos(FtF)
        G *= np.sqrt(num / den)
        X_SMNF = G @ F.T
        error = np.linalg.norm(X - X_SMNF)
        errors.append(error)
    return F, G, np.array(errors), X_SMNF
