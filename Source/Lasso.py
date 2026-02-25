""" CODE AUTHORED BY LUIS IGLESIAS MARTINEZ """
import numpy as np
from tqdm import tqdm
import random

def soft_thresholding(z, y):
    return np.sign(z) * np.maximum(np.abs(z) - y, 0.0)


def project_partial_residual(X, B_j, R, index):
    pr = np.mean((R + X[:,index:index+1]*B_j)*X[:,index:index+1])
    return pr


def estimate_lmax_lmin(X,Y, a = 1., e = 0.01):
    N = X.shape[0]
    # Project X onto Y
    proj = np.matmul(X.T,Y)
    # Get Maxima
    maxl = np.max(np.abs(proj), axis = 0)
    lmax = maxl/N*a
    lmin = e*lmax
    return lmax, lmin, proj


def preprocessing(X, Y):
    # unit normalize and zero centre
    M = np.mean(X, axis = 0, keepdims = True)
    S = np.sum(X**2, axis = 0, keepdims = True)/X.shape[0] - M**2
    X_n = (X - M)/S**.5
    # Remove mean of y
    Y_n = Y - np.mean(Y, axis = 0, keepdims = True)
    return X_n, Y_n, M, S


def coordinate_descent_step(XTY, G, B, a,  lmax, lmin, k, K):
    """
    Vectorized coordinate descent update.
    XTY : (p, q) precomputed X.T @ Y / n
    G   : (p, p) precomputed X.T @ X / n
    B   : (p, q) coefficient matrix
    lmax   : (q,) penalty per output
    lmin : (q, )
    k : integer (optimization step)
    """
    # Calculate the Projected Residuals
    PR = XTY - G@B
    l = lmax*(lmin/lmax)**(k/(K-1))
    # Strong Rule
    if k < (K - 1):
        l_1 = lmax*(lmin/lmax)**((k+1)/(K-1))
        Active_Set = np.any(np.abs(PR) > 2*l - l_1, axis=1)
    else:
        Active_Set = np.ones(PR.shape, dtype=bool)
    B_new = B.copy()
    p, q = B.shape
    for i in range(p):
        if not Active_Set[i].any():
            continue
        # vector of pseudoresiduals across all outputs
        pr = PR[i,:] + G[i, i] * B[i, :]
        # soft threshold + update all outputs at once
        B_new[i, :] = soft_thresholding(pr, l * a) / (1 + l * (1 - a))
        # recompute pr
        # update B
        PR -= np.outer(G[:,i], B_new[i,:] - B[i,:])
    return B_new



def train(X, Y,  lmax, lmin, a = 1,  K=100, tol=1e-4, max_ite=100):
    X_n, Y_n, M, S = preprocessing(X, Y)
    n = X_n.shape[0]
    # Precompute Gram matrices
    G = (X_n.T @ X_n) / n          # (p, p)
    XTY = (X_n.T @ Y_n) / n        # (p, q)
    B = np.zeros((X_n.shape[1], Y_n.shape[1], K))
    for k in range(K):
        if k > 0:
            B[:, :, k] = B[:, :, k-1]
        converged = False
        counter = 0
        while not converged:
            counter += 1
            B_new = coordinate_descent_step(XTY, G, B[:,:,k], a, lmax, lmin, k, K)
            change = np.mean((B[:, :, k] - B_new)**2)
            B[:, :, k] = B_new
            if change < tol or counter >= max_ite:
                converged = True
    return B, M, S


def cross_validation(X, Y, a=1, e=0.01, Kfold=10, K=100, tol=1e-4, max_ite=100, seed=42):
    n = X.shape[0]
    indexes = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indexes)
    val_chunk = n // Kfold
    X_n, Y_n, M, S = preprocessing(X, Y)
    # Global lmax/lmin
    lmax, lmin, proj = estimate_lmax_lmin(X_n, Y_n, a, e)
    # Store CV errors
    MSE = np.zeros((Y.shape[1], K, Kfold))
    pbar = tqdm(range(Kfold))
    for i in pbar:
        # Build folds
        val_set = indexes[i * val_chunk: (i + 1) * val_chunk] if i < Kfold - 1 else indexes[i * val_chunk:]
        train_set = np.setdiff1d(indexes, val_set)
        X_train, Y_train = X_n[train_set], Y_n[train_set]
        X_val, Y_val = X_n[val_set], Y_n[val_set]
        # Train on this fold
        B, M, S = train(X_train, Y_train,  lmax, lmin, a = a, K=K, tol=tol, max_ite=max_ite)
        # normalize X_val
        X_val = (X_val-M)/S
        # Validation MSE
        for j in range(K):
            F_val = X_val @ B[:, :, j]
            MSE[:, j, i] = np.mean((F_val - Y_val) ** 2, axis=0)
        pbar.set_postfix({'fold': i})
    # Select best lambda
    avg_mse = np.mean(MSE, axis=2)  # shape (q, K)
    top_lambda_idx = np.argmin(avg_mse, axis=1)
    # Retrain full model
    B_full, M, S = train(X, Y, lmax, lmin, a, K, tol, max_ite)
    B_best = [B_full[:, j, top_lambda_idx[j]] for j in range(len(top_lambda_idx))]
    return B_best, MSE, B_full


