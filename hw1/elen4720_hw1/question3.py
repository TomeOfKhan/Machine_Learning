"""ELEN 4720 Homework 1, Question 3: Parts 1 and 2.

Run: python question3.py
Dependencies: NumPy and Matplotlib.
Place the four extracted CSV files in data/ beside this file.
Part 1: (a)-(c), linear ridge regression.
Part 2: (d), polynomial ridge regression using Lecture 2, page 28.
All additional columns use training-only standardization (Lecture 3, page 25).
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

def run_part1():
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


# Part 2: polynomial regression
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

def run_part2():
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


def main():
    run_part1()
    run_part2()

if __name__ == '__main__':
    main()
