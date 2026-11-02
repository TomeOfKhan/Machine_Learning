from pathlib import Path
import json
import os

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
import numpy as np
from scipy.special import expit, logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
SEED = 4720
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 220})


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)


def confusion(y, prediction):

    return np.bincount(2*y.astype(int) + prediction.astype(int),
                       minlength=4).reshape(2, 2)


def logistic_objective(A, y, w):
    return -np.logaddexp(0., -y*(A@w)).sum()


def logistic_derivatives(A, y, w):
    margins = y*(A@w)
    residual = expit(-margins)
    gradient = A.T @ (y*residual)

    curvature = expit(margins)*residual
    negative_hessian = A.T @ (curvature[:, None]*A)
    return gradient, negative_hessian


def steepest_ascent(A, y, iterations=1000):
    w = np.zeros(A.shape[1])
    history = np.empty(iterations)
    eta = 0.01/4600
    for t in range(iterations):
        gradient = A.T @ (y*expit(-y*(A@w)))
        w += eta*gradient
        history[t] = logistic_objective(A, y, w)
    return w, history


def newton_ascent(A, y, iterations=100):
    w = np.zeros(A.shape[1])
    history = np.empty(iterations)
    for t in range(iterations):
        gradient, negative_hessian = logistic_derivatives(A, y, w)


        delta = np.linalg.lstsq(negative_hessian, gradient, rcond=None)[0]
        w += delta
        history[t] = logistic_objective(A, y, w)
    return w, history


def verify_derivatives(A, y):

    rng = np.random.default_rng(17)
    B, labels = A[:120], y[:120]
    w = rng.normal(size=B.shape[1])*1e-3
    direction = rng.normal(size=B.shape[1])
    direction /= np.linalg.norm(direction)
    h = 1e-6
    gradient, negative_hessian = logistic_derivatives(B, labels, w)
    finite_gradient = (logistic_objective(B, labels, w+h*direction)
                       - logistic_objective(B, labels, w-h*direction))/(2*h)
    np.testing.assert_allclose(finite_gradient, gradient@direction, rtol=2e-6, atol=1e-5)
    plus, _ = logistic_derivatives(B, labels, w+h*direction)
    minus, _ = logistic_derivatives(B, labels, w-h*direction)
    np.testing.assert_allclose((plus-minus)/(2*h), -negative_hessian@direction,
                               rtol=2e-5, atol=1e-4)


def problem2():
    X = np.loadtxt(ROOT/'data/Bayes_classifier/X.csv', delimiter=',')
    y = np.loadtxt(ROOT/'data/Bayes_classifier/y.csv', delimiter=',').astype(int)
    assert X.shape == (4600, 54) and set(np.unique(y)) == {0, 1}
    assert np.all(X >= 0) and np.isfinite(X).all()
    A = np.column_stack([X, np.ones(4600)])
    signed_y = 2*y-1
    verify_derivatives(A, signed_y)


    groups = np.array_split(np.random.default_rng(SEED).permutation(4600), 10)
    fold_id = np.empty(4600, dtype=int)
    nb_prediction, newton_prediction = np.empty(4600, int), np.empty(4600, int)
    nb_scores = np.empty((4600, 2))
    newton_scores = np.empty(4600)
    parameters = np.empty((10, 2, 54))
    priors = np.empty(10)
    steep_history, newton_history = np.empty((10, 1000)), np.empty((10, 100))
    newton_weights, steep_weights = np.empty((10, 55)), np.empty((10, 55))
    gradient_norms = []
    for fold, test in enumerate(groups):
        train = np.ones(4600, dtype=bool)
        train[test] = False
        fold_id[test] = fold+1
        x_train, y_train = X[train], y[train]
        pi = y_train.mean()
        priors[fold] = pi
        for label in (0, 1):
            class_rows = x_train[y_train == label]
            parameters[fold, label] = (1+class_rows.sum(axis=0))/(1+len(class_rows))

        scores = (X[test] @ np.log(parameters[fold]).T
                  - parameters[fold].sum(axis=1)
                  + np.log([1-pi, pi]))
        nb_scores[test] = scores
        nb_prediction[test] = np.argmax(scores, axis=1)
        steep_weights[fold], steep_history[fold] = steepest_ascent(A[train], signed_y[train])
        newton_weights[fold], newton_history[fold] = newton_ascent(A[train], signed_y[train])
        newton_scores[test] = A[test] @ newton_weights[fold]
        newton_prediction[test] = (newton_scores[test] >= 0).astype(int)
        gradient, _ = logistic_derivatives(A[train], signed_y[train], newton_weights[fold])
        gradient_norms.append(float(np.linalg.norm(gradient, ord=np.inf)))
        assert np.isfinite(steep_history[fold]).all() and np.isfinite(newton_history[fold]).all()
        assert np.all(np.diff(newton_history[fold]) >= -1e-7)
        print(f'Problem 2 fold {fold+1}/10: Newton L={newton_history[fold,-1]:.6f}', flush=True)

    nb_cm, nr_cm = confusion(y, nb_prediction), confusion(y, newton_prediction)
    assert nb_cm.sum() == 4600 and nr_cm.sum() == 4600
    assert all(np.sum(fold_id == i) == 460 for i in range(1, 11))
    mean_parameters = parameters.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8.7, 4.4))
    dimensions = np.arange(1, 55)
    ax.stem(dimensions-.12, mean_parameters[0], linefmt='C0-', markerfmt='C0o',
            basefmt=' ', label='Non-spam (y=0)')
    ax.stem(dimensions+.12, mean_parameters[1], linefmt='C1-', markerfmt='C1s',
            basefmt=' ', label='Spam (y=1)')
    ax.set(xlabel='Feature dimension', ylabel='Mean Poisson MAP parameter', xlim=(0, 55))
    ax.set_xticks([1, 5, 10, 16, 20, 25, 30, 35, 40, 45, 50, 52, 54])
    ax.legend(); ax.grid(axis='y', alpha=.25)
    save_figure(fig, 'p2b_poisson_parameters.png')
    for histories, name, xlabel in [(steep_history, 'p2c_steepest_ascent.png', 'Iteration (1-1000)'),
                                    (newton_history, 'p2d_newton.png', 'Iteration (1-100)')]:
        fig, ax = plt.subplots(figsize=(8.7, 4.5))
        for fold, values in enumerate(histories):
            ax.plot(np.arange(1, len(values)+1), values, label=f'Fold {fold+1}', linewidth=1)
        ax.set(xlabel=xlabel, ylabel='Training log likelihood L(w)')
        ax.legend(ncol=5, fontsize=8); ax.grid(alpha=.25)
        save_figure(fig, name)
    np.savetxt(OUT/'p2_predictions.csv', np.column_stack([np.arange(1, 4601), fold_id,
                y, nb_prediction, newton_prediction, newton_scores]), delimiter=',',
                header='row,held_out_fold,true_label,naive_bayes,newton,newton_score', comments='')
    np.savetxt(OUT/'p2_mean_poisson_parameters.csv', np.column_stack([dimensions, mean_parameters.T]),
               delimiter=',', header='dimension,nonspam,spam', comments='')
    np.savetxt(OUT/'p2c_objectives.csv', np.column_stack([np.arange(1,1001), steep_history.T]),
               delimiter=',', header='iteration,'+','.join(f'fold{i}' for i in range(1,11)), comments='')
    np.savetxt(OUT/'p2d_objectives.csv', np.column_stack([np.arange(1,101), newton_history.T]),
               delimiter=',', header='iteration,'+','.join(f'fold{i}' for i in range(1,11)), comments='')
    np.savez(OUT/'p2_models.npz', poisson_parameters=parameters, class_priors=priors,
             newton_weights=newton_weights, steepest_weights=steep_weights, nb_scores=nb_scores)
    return {'seed': SEED, 'fold_sizes': [len(g) for g in groups],
            'naive_bayes_confusion': nb_cm.tolist(), 'naive_bayes_accuracy': float(np.trace(nb_cm)/4600),
            'newton_confusion': nr_cm.tolist(), 'newton_accuracy': float(np.trace(nr_cm)/4600),
            'mean_parameter_dimension16': mean_parameters[:,15].tolist(),
            'mean_parameter_dimension52': mean_parameters[:,51].tolist(),
            'steepest_initial_objective': float(-4140*np.log(2)),
            'steepest_final_objectives': steep_history[:,-1].tolist(),
            'newton_final_objectives': newton_history[:,-1].tolist(),
            'newton_final_gradient_inf_norm': gradient_norms}


def problem3():
    X = np.loadtxt(ROOT/'data/boosting/X.csv', delimiter=',')
    y = np.loadtxt(ROOT/'data/boosting/y.csv', delimiter=',').astype(int)
    n, d = X.shape
    assert (n, d) == (1000, 3) and np.all(X[:,-1] == 1)
    assert set(np.unique(y)) == {-1, 1}
    T = 2500
    rng = np.random.default_rng(SEED+1)
    log_distribution = np.full(n, -np.log(n))
    distribution_sum = np.zeros(n)
    ensemble_score = np.zeros(n)
    errors, alphas, train_errors, log_bounds = [np.empty(T) for _ in range(4)]
    weak_weights = np.empty((T, d))
    log_bound = 0.
    flips = 0
    for t in range(T):
        distribution = np.exp(log_distribution)
        distribution /= distribution.sum()
        distribution_sum += distribution
        bootstrap = rng.choice(n, size=n, replace=True, p=distribution)
        beta = np.linalg.lstsq(X[bootstrap], y[bootstrap], rcond=None)[0]
        prediction = np.where(X@beta >= 0, 1, -1)
        epsilon = float(distribution[prediction != y].sum())
        if epsilon > .5:
            beta = -beta
            prediction = np.where(X@beta >= 0, 1, -1)
            epsilon = float(distribution[prediction != y].sum())
            flips += 1
        if not 0 < epsilon < 1:
            raise ArithmeticError('A perfect weak learner needs an early-stop convention.')
        alpha = .5*np.log((1-epsilon)/epsilon)
        ensemble_score += alpha*prediction
        errors[t], alphas[t] = epsilon, alpha
        weak_weights[t] = beta
        train_errors[t] = np.mean(np.where(ensemble_score >= 0, 1, -1) != y)
        unnormalized = log_distribution-alpha*y*prediction
        log_Z = logsumexp(unnormalized)
        log_distribution = unnormalized-log_Z

        expected_log_Z = np.log(2)+.5*(np.log(epsilon)+np.log1p(-epsilon))
        np.testing.assert_allclose(log_Z, expected_log_Z, atol=1e-12)
        log_bound += expected_log_Z
        log_bounds[t] = log_bound
        assert train_errors[t] <= np.exp(log_bound)+1e-12
        if (t+1) % 500 == 0:
            print(f'Problem 3 round {t+1}/{T}: error={train_errors[t]:.4f}', flush=True)
    average_distribution = distribution_sum/T
    np.testing.assert_allclose(average_distribution.sum(), 1, atol=1e-12)
    np.testing.assert_allclose(log_bounds[-1], logsumexp(-y*ensemble_score)-np.log(n), atol=1e-10)
    fig, ax = plt.subplots(figsize=(8.7, 4.4))
    iterations = np.arange(1, T+1)
    ax.plot(iterations, train_errors, label='Empirical training error')
    ax.plot(iterations, np.exp(log_bounds), label='Exponential-loss upper bound')
    edge_bound = np.exp(-2*np.cumsum((.5-errors)**2))
    ax.plot(iterations, edge_bound, '--', label='Edge-based upper bound')
    ax.set(xlabel='Boosting round t', ylabel='Training error / upper bound', ylim=(0, 1.02))
    ax.legend(); ax.grid(alpha=.25); save_figure(fig, 'p3a_training_error_bound.png')
    fig, ax = plt.subplots(figsize=(8.7, 4.4))
    markerline, stemlines, baseline = ax.stem(np.arange(1,n+1), average_distribution,
                                             linefmt='C0-', markerfmt=' ', basefmt=' ')
    stemlines.set_linewidth(.65)
    ax.set(xlabel='Training observation index', ylabel='Average sampling probability')
    ax.grid(axis='y', alpha=.25); save_figure(fig, 'p3b_average_distribution.png')
    for values, ylabel, name in [(errors, r'$\epsilon_t$', 'p3c_epsilon.png'),
                                  (alphas, r'$\alpha_t$', 'p3c_alpha.png')]:
        fig, ax = plt.subplots(figsize=(8.7, 4.1))
        ax.plot(iterations, values, linewidth=.6)
        ax.set(xlabel='Boosting round t', ylabel=ylabel)
        ax.grid(alpha=.25); save_figure(fig, name)
    np.savetxt(OUT/'p3_rounds.csv', np.column_stack([iterations, errors, alphas, train_errors,
                np.exp(log_bounds), edge_bound, weak_weights]), delimiter=',',
                header='round,epsilon,alpha,train_error,upper_bound,edge_upper_bound,beta1,beta2,intercept', comments='')
    np.savetxt(OUT/'p3_average_distribution.csv', np.column_stack([np.arange(1,n+1), average_distribution]),
               delimiter=',', header='row,average_probability', comments='')
    np.savetxt(OUT/'p3_predictions.csv', np.column_stack([y, ensemble_score,
                np.where(ensemble_score >= 0, 1, -1)]), delimiter=',',
                header='true_label,ensemble_score,predicted_label', comments='')
    return {'seed': SEED+1, 'rounds': T, 'sample_count': n,
            'final_training_error': float(train_errors[-1]), 'final_training_mistakes': int(round(n*train_errors[-1])),
            'final_upper_bound': float(np.exp(log_bounds[-1])), 'sign_flips': flips,
            'final_edge_upper_bound': float(edge_bound[-1]),
            'epsilon_min': float(errors.min()), 'epsilon_max': float(errors.max()),
            'alpha_min': float(alphas.min()), 'alpha_max': float(alphas.max()),
            'maximum_average_probability': float(average_distribution.max()),
            'most_weighted_observation': int(np.argmax(average_distribution)+1)}


def main():
    OUT.mkdir(exist_ok=True)
    results = {'problem2': problem2(), 'problem3': problem3()}
    (OUT/'summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
