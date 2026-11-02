"""ELEN 4720 HW1: run with Python, NumPy and Matplotlib; uses extracted CSVs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
X = np.loadtxt(ROOT / 'data/X_train.csv', delimiter=',')
Xt = np.loadtxt(ROOT / 'data/X_test.csv', delimiter=',')
y = np.loadtxt(ROOT / 'data/y_train.csv', delimiter=',').reshape(-1)
yt = np.loadtxt(ROOT / 'data/y_test.csv', delimiter=',').reshape(-1)
assert X.shape == (350, 7) and Xt.shape == (42, 7)
assert np.all(X[:, -1] == 1) and np.all(Xt[:, -1] == 1)

def ridge_path(A, target, lambdas):
    """Penalize every coefficient, including intercept, as specified."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    ls = np.asarray(lambdas, dtype=float)
    factors = s[:, None] / (s[:, None] ** 2 + ls[None, :])
    if np.any(ls == 0):
        tol = max(A.shape) * np.finfo(float).eps * s[0]
        factors[:, ls == 0] = np.divide(1, s, out=np.zeros_like(s), where=s > tol)[:, None]
    return Vt.T @ (factors * (U.T @ target)[:, None]), np.sum(s[:, None]**2 / (s[:, None]**2 + ls), axis=0)

def polynomial(p):
    """Separate feature powers, no interactions; keep supplied linear columns."""
    train, test = [X[:, :6]], [Xt[:, :6]]
    for power in range(2, p + 1):
        a, b = X[:, :6] ** power, Xt[:, :6] ** power
        mean, std = a.mean(axis=0), a.std(axis=0, ddof=0)
        std = np.where(std == 0, 1, std)
        train.append((a - mean) / std)
        test.append((b - mean) / std)
    return np.column_stack(train + [X[:, -1]]), np.column_stack(test + [Xt[:, -1]])

labels = ['1: cylinders', '2: displacement', '3: horsepower', '4: weight', '5: acceleration', '6: year made', '7: intercept']
lam = np.arange(5001)
w, df = ridge_path(X, y, lam)
fig, ax = plt.subplots(figsize=(8, 4.7))
for i, label in enumerate(labels):
    ax.plot(df, w[i], label=label)
ax.set(xlabel='Effective degrees of freedom df(lambda)', ylabel='Coefficient', title='Ridge coefficient paths')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / 'coefficient_paths.png', dpi=180); plt.close(fig)
lam = np.arange(51)
w50, _ = ridge_path(X, y, lam)
pred = Xt @ w50
rmse = np.sqrt(np.mean((yt[:, None] - pred) ** 2, axis=0))
np.savetxt(OUT / 'linear_test_predictions.csv', pred, delimiter=',', header=','.join(f'lambda_{i}' for i in lam), comments='')
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lam, rmse); ax.set(xlabel='lambda', ylabel='Test RMSE (mpg)', title='Linear ridge regression'); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / 'linear_rmse.png', dpi=180); plt.close(fig)
summary = {'train_shape': list(X.shape), 'test_shape': list(Xt.shape), 'linear_best_lambda_0_50': int(np.argmin(rmse)), 'linear_best_rmse_0_50': float(rmse.min()), 'ls_rmse': float(rmse[0]), 'ls_weights': w50[:, 0].tolist(), 'polynomial': {}}
lam = np.arange(101)
fig, ax = plt.subplots(figsize=(8, 4.3))
for p in (1, 2, 3):
    A, B = polynomial(p)
    weights, _ = ridge_path(A, y, lam)
    errors = np.sqrt(np.mean((yt[:, None] - B @ weights)**2, axis=0))
    ax.plot(lam, errors, label=f'p = {p}')
    summary['polynomial'][str(p)] = {'best_lambda': int(np.argmin(errors)), 'best_rmse': float(errors.min()), 'lambda_0_rmse': float(errors[0]), 'features': A.shape[1]}
    np.savetxt(OUT / f'rmse_p{p}.csv', np.column_stack([lam, errors]), delimiter=',', header='lambda,test_rmse', comments='')
ax.set(xlabel='lambda', ylabel='Test RMSE (mpg)', title='Polynomial ridge regression'); ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / 'polynomial_rmse.png', dpi=180); plt.close(fig)
# Verify SVD computations against independent direct solutions.
np.testing.assert_allclose(w50[:, 0], np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-10)
np.testing.assert_allclose(w50[:, 10], np.linalg.solve(X.T @ X + 10*np.eye(7), X.T @ y), atol=1e-10)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
