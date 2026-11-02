"""Question 3, Part 2 (d). Run: python question3_part2.py.

Modifies Part 1 with separate powers of the six features, p=1,2,3.
Uses training statistics to standardize every additional dimension.
Uses the separate feature powers in Lecture 2, page 28 (6p+1 columns),
and the population-standard-deviation formula in Lecture 3, page 25.
Requires question3_part1.py alongside this file for the common SVD solver.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from question3_part1 import ridge_path

ROOT = Path(__file__).resolve().parent

def polynomial_features(X_train, X_test, p):
    """6p features plus one constant; fit transforms on training data only."""
    train_columns, test_columns = [X_train[:, :6]], [X_test[:, :6]]
    statistics = []
    for power in range(2, p+1):
        train_power = X_train[:, :6]**power
        test_power = X_test[:, :6]**power
        mean = train_power.mean(axis=0)
        std = train_power.std(axis=0, ddof=0)
        std = np.where(std == 0, 1., std)
        train_columns.append((train_power - mean)/std)
        test_columns.append((test_power - mean)/std)
        statistics.append({'power': power, 'training_mean': mean.tolist(),
                           'training_std': std.tolist()})
    # Do not exponentiate or standardize the constant column.
    train_columns.append(np.ones((X_train.shape[0], 1)))
    test_columns.append(np.ones((X_test.shape[0], 1)))
    return np.column_stack(train_columns), np.column_stack(test_columns), statistics

def main():
    out = ROOT/'results/part2'
    out.mkdir(parents=True, exist_ok=True)
    X = np.loadtxt(ROOT/'data/X_train.csv', delimiter=',')
    y = np.loadtxt(ROOT/'data/y_train.csv', delimiter=',').reshape(-1)
    X_test = np.loadtxt(ROOT/'data/X_test.csv', delimiter=',')
    y_test = np.loadtxt(ROOT/'data/y_test.csv', delimiter=',').reshape(-1)
    assert X.shape == (350, 7) and X_test.shape == (42, 7)
    lambdas = np.arange(0, 101)
    summary, curves = {}, []
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in (1, 2, 3):
        A, B, stats = polynomial_features(X, X_test, p)
        assert A.shape == (350, 6*p+1) and B.shape == (42, 6*p+1)
        if p > 1:
            np.testing.assert_allclose(A[:, 6:-1].mean(axis=0), 0, atol=1e-12)
            np.testing.assert_allclose(A[:, 6:-1].std(axis=0), 1, atol=1e-12)
        weights, _ = ridge_path(A, y, lambdas)
        predictions = B @ weights
        errors = np.sqrt(np.mean((y_test[:, None] - predictions)**2, axis=0))
        curves.append(errors)
        ax.plot(lambdas, errors, label=f'p = {p}')
        np.savetxt(out/f'predictions_p{p}.csv', predictions, delimiter=',',
                   header=','.join(f'lambda_{i}' for i in lambdas), comments='')
        np.savetxt(out/f'coefficients_p{p}.csv', np.column_stack([lambdas, weights.T]),
                   delimiter=',', header='lambda,'+','.join(f'w{i+1}' for i in range(A.shape[1])), comments='')
        np.testing.assert_allclose(weights[:, 0], np.linalg.lstsq(A, y, rcond=None)[0],
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(weights[:, 50],
                                   np.linalg.solve(A.T@A + 50*np.eye(A.shape[1]), A.T@y),
                                   rtol=1e-9, atol=1e-9)
        summary[str(p)] = {'features': A.shape[1], 'best_lambda': int(np.argmin(errors)),
                           'best_rmse': float(errors.min()), 'lambda_0_rmse': float(errors[0]),
                           'standardization': stats}
    ax.set(xlabel=r'$\lambda$', ylabel='Test RMSE (mpg)')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out/'d_rmse_vs_lambda.png', dpi=180); plt.close(fig)
    np.savetxt(out/'d_rmse.csv', np.column_stack([lambdas]+curves), delimiter=',',
               header='lambda,rmse_p1,rmse_p2,rmse_p3', comments='')
    # p=1 must reproduce Part 1(c) exactly on their common lambda grid.
    base_weights, _ = ridge_path(X, y, np.arange(51))
    base_errors = np.sqrt(np.mean((y_test[:, None] - X_test@base_weights)**2, axis=0))
    np.testing.assert_allclose(curves[0][:51], base_errors, rtol=1e-12, atol=1e-12)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    for p, result in summary.items():
        print(f"Part 2(d): p={p}, best lambda={result['best_lambda']}, RMSE={result['best_rmse']:.6f}")

if __name__ == '__main__':
    main()
