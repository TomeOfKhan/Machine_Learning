"""Question 3, Part 1 (a)-(c). Run: python question3_part1.py.

Uses only training data to estimate coefficients; test data are used for (c).
The extracted CSV files must be in data/ beside this source file.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LABELS = ['1: cylinders', '2: displacement', '3: horsepower',
          '4: weight', '5: acceleration', '6: year made', '7: constant']

def ridge_path(X, y, lambdas):
    """SVD solution of ||y-Xw||^2 + lambda ||w||^2, including constant."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lambdas = np.asarray(lambdas, dtype=float)
    gains = s[:, None] / (s[:, None]**2 + lambdas[None, :])
    zero = lambdas == 0
    if np.any(zero):
        tolerance = max(X.shape) * np.finfo(float).eps * s[0]
        gains[:, zero] = np.divide(1., s, out=np.zeros_like(s),
                                  where=s > tolerance)[:, None]
    weights = Vt.T @ (gains * (U.T @ y)[:, None])
    df = np.sum(s[:, None]**2 / (s[:, None]**2 + lambdas[None, :]), axis=0)
    return weights, df

def main():
    out = ROOT / 'results' / 'part1'
    out.mkdir(parents=True, exist_ok=True)
    X = np.loadtxt(ROOT/'data/X_train.csv', delimiter=',')
    y = np.loadtxt(ROOT/'data/y_train.csv', delimiter=',').reshape(-1)
    assert X.shape == (350, 7) and y.shape == (350,)
    assert np.all(X[:, 6] == 1)

    # (a): train on all and only the 350 training observations.
    lambdas = np.arange(0, 5001)
    weights, df = ridge_path(X, y, lambdas)
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, label in enumerate(LABELS):
        ax.plot(df, weights[j], label=label)
    ax.set(xlabel=r'$df(\lambda)$', ylabel='Ridge coefficient')
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'a_coefficients_vs_df.png', dpi=180)
    plt.close(fig)
    np.savetxt(out/'a_coefficients.csv', np.column_stack([lambdas, df, weights.T]),
               delimiter=',', header='lambda,df,w1,w2,w3,w4,w5,w6,w7', comments='')

    # (b): report the least-squares coefficients for the interpretation.
    print('Part 1(b): coefficient dimensions 1 through 7:', weights[:, 0])
    print('Dimensions 4 (weight) and 6 (year made) stand out.')

    # (c): test evaluation is separate from training.
    X_test = np.loadtxt(ROOT/'data/X_test.csv', delimiter=',')
    y_test = np.loadtxt(ROOT/'data/y_test.csv', delimiter=',').reshape(-1)
    assert X_test.shape == (42, 7) and y_test.shape == (42,)
    assert np.all(X_test[:, 6] == 1)
    lambdas_test = np.arange(0, 51)
    predictions = X_test @ weights[:, :51]
    errors = np.sqrt(np.mean((y_test[:, None] - predictions)**2, axis=0))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lambdas_test, errors)
    ax.set(xlabel=r'$\lambda$', ylabel='Test RMSE (mpg)')
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out/'c_rmse_vs_lambda.png', dpi=180); plt.close(fig)
    np.savetxt(out/'c_predictions.csv', predictions, delimiter=',',
               header=','.join(f'lambda_{i}' for i in lambdas_test), comments='')
    np.savetxt(out/'c_rmse.csv', np.column_stack([lambdas_test, errors]),
               delimiter=',', header='lambda,test_rmse', comments='')
    # Independent checks of the SVD formula against standard linear solvers.
    np.testing.assert_allclose(weights[:, 0], np.linalg.lstsq(X, y, rcond=None)[0],
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(weights[:, 10],
                               np.linalg.solve(X.T@X + 10*np.eye(7), X.T@y),
                               rtol=1e-10, atol=1e-10)
    summary = {'best_lambda': int(lambdas_test[np.argmin(errors)]),
               'best_rmse': float(errors.min()), 'least_squares_rmse': float(errors[0]),
               'least_squares_weights': weights[:, 0].tolist()}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('Part 1(c):', json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
