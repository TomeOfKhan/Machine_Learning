from pathlib import Path
import argparse
import csv
import json
import os
import time

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'hw3-data'
OUT = ROOT / 'results'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 220})


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches='tight')
    plt.close(fig)


def factor_update(other, groups, regularizer):
    result = np.empty((len(groups), other.shape[1]))
    for i, (indices, values) in enumerate(groups):
        factors = other[indices]
        result[i] = np.linalg.solve(factors.T @ factors + regularizer,
                                    factors.T @ values)
    return result


def grouped_observations(ids, other_ids, ratings, count):
    order = np.argsort(ids, kind='stable')
    boundaries = np.searchsorted(ids[order], np.arange(count+1))
    return [(other_ids[order[boundaries[i]:boundaries[i+1]]],
             ratings[order[boundaries[i]:boundaries[i+1]]]) for i in range(count)]


def problem1():
    train = np.loadtxt(DATA/'ratings_train.csv', delimiter=',')
    test = np.loadtxt(DATA/'ratings_test.csv', delimiter=',')
    titles = (DATA/'movies.txt').read_text(encoding='utf-8-sig').splitlines()
    user_ids, movie_ids = train[:, 0].astype(int)-1, train[:, 1].astype(int)-1
    test_users, test_movies = test[:, 0].astype(int)-1, test[:, 1].astype(int)-1
    ratings = train[:, 2]
    n_users = int(max(user_ids.max(), test_users.max())+1)
    n_movies = len(titles)
    assert train.shape == (95000, 3) and test.shape == (5000, 3)
    assert n_users == 943 and n_movies == 1682
    assert movie_ids.max() < n_movies and test_movies.max() < n_movies
    user_groups = grouped_observations(user_ids, movie_ids, ratings, n_users)
    movie_groups = grouped_observations(movie_ids, user_ids, ratings, n_movies)
    variance, dimension, precision = .25, 10, 1.
    regularizer = variance*precision*np.eye(dimension)
    constant = (-len(ratings)/2*np.log(2*np.pi*variance)
                +(n_users+n_movies)*dimension/2*np.log(precision/(2*np.pi)))
    trajectories = np.empty((100, 10))
    rows, best_factors = [], None
    best_objective = -np.inf
    for run in range(10):
        started = time.perf_counter()
        rng = np.random.default_rng(4720+run)
        U = rng.normal(size=(n_users, dimension))
        V = rng.normal(size=(n_movies, dimension))
        for t in range(100):
            U = factor_update(V, user_groups, regularizer)
            V = factor_update(U, movie_groups, regularizer)
            predictions = np.einsum('ij,ij->i', U[user_ids], V[movie_ids])
            reduced = (-np.sum((ratings-predictions)**2)/(2*variance)
                       -precision/2*(np.sum(U*U)+np.sum(V*V)))
            trajectories[t, run] = constant+reduced
        assert np.all(np.diff(trajectories[:, run]) >= -1e-7)
        predictions_test = np.einsum('ij,ij->i', U[test_users], V[test_movies])
        rmse = float(np.sqrt(np.mean((test[:, 2]-predictions_test)**2)))
        objective = float(trajectories[-1, run])
        rows.append({'run': run+1, 'seed': 4720+run, 'log_joint': objective,
                     'objective_without_constant': float(reduced), 'test_rmse': rmse})
        indices, values = movie_groups[0]
        np.testing.assert_allclose((U[indices].T@U[indices]+regularizer)@V[0],
                                   U[indices].T@values, rtol=1e-9, atol=1e-9)
        if objective > best_objective:
            best_objective = objective
            best_factors = (U.copy(), V.copy(), predictions_test.copy(), run+1)
        print(f'PMF run {run+1}/10: log joint={objective:.6f}, test RMSE={rmse:.6f}, '
              f'{time.perf_counter()-started:.1f}s', flush=True)
    rows.sort(key=lambda item: item['log_joint'], reverse=True)
    with (OUT/'p1_final_results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(OUT/'p1_objectives.csv', np.column_stack([np.arange(1, 101), trajectories]),
               delimiter=',', header='iteration,'+','.join(f'run{i}' for i in range(1, 11)), comments='')
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for run in range(10):
        ax.plot(np.arange(2, 101), trajectories[1:, run], linewidth=1, label=f'Run {run+1}')
    ax.set(xlabel='Iteration', ylabel='Training log joint likelihood', xlim=(2, 100))
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.legend(ncol=5, fontsize=8); ax.grid(alpha=.25)
    save_figure(fig, 'p1a_log_joint.png')
    U, V, predictions_test, best_run = best_factors
    neighbors = []
    for query in ('Star Wars', 'My Fair Lady', 'Goodfellas'):
        candidates = [i for i, title in enumerate(titles) if title.casefold().startswith(query.casefold()+' (')]
        assert len(candidates) == 1, (query, candidates)
        index = candidates[0]
        distances = np.linalg.norm(V-V[index], axis=1)
        distances[index] = np.inf
        nearest = np.argsort(distances, kind='stable')[:10]
        neighbors.append({'query': titles[index], 'movie_id': index+1,
                          'neighbors': [{'movie_id': int(i+1), 'title': titles[i],
                                         'distance': float(distances[i])} for i in nearest]})
    np.savez_compressed(OUT/'p1_best_model.npz', U=U, V=V, run=best_run)
    np.savetxt(OUT/'p1_test_predictions.csv', np.column_stack([test, predictions_test]),
               delimiter=',', header='user_id,movie_id,observed_centered_rating,predicted_centered_rating', comments='')
    result = {'sigma_squared': variance, 'lambda': precision, 'dimension': dimension,
              'iterations': 100, 'log_joint_constant': float(constant),
              'sorted_results': rows, 'best_run': best_run, 'neighbors': neighbors}
    (OUT/'p1_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def load_news():
    vocabulary = (DATA/'nyt_vocab.dat').read_text(encoding='utf-8-sig').splitlines()
    documents = (DATA/'nyt_data.txt').read_text(encoding='utf-8-sig').splitlines()
    rows, columns, counts = [], [], []
    for j, line in enumerate(documents):
        for entry in line.split(','):
            index, count = entry.split(':')
            rows.append(int(index)-1)
            columns.append(j)
            counts.append(float(count))
    X = sparse.coo_matrix((counts, (rows, columns)),
                          shape=(len(vocabulary), len(documents))).tocsr()
    X.sum_duplicates()
    X.eliminate_zeros()
    assert X.shape == (3012, 8447) and np.all(X.data > 0)
    return X, vocabulary


def observed_predictions(W, H, rows, columns):
    predictions = np.empty(len(rows))
    for start in range(0, len(rows), 40000):
        end = start+40000
        predictions[start:end] = np.einsum('ij,ij->i',
            W[rows[start:end]], H[:, columns[start:end]].T)
    return predictions


def problem2():
    started = time.perf_counter()
    X, vocabulary = load_news()
    coo = X.tocoo()
    rows, columns, counts = coo.row, coo.col, coo.data
    rng = np.random.default_rng(5720)
    W = rng.uniform(1, 2, size=(X.shape[0], 25))
    H = rng.uniform(1, 2, size=(25, X.shape[1]))
    tiny = 1e-16
    history = np.empty(100)
    count_sum = counts.sum()
    fixed_term = np.sum(counts*np.log(counts))-count_sum
    for t in range(100):
        prediction = observed_predictions(W, H, rows, columns)
        ratio = sparse.csr_matrix((counts/(prediction+tiny), (rows, columns)), shape=X.shape)
        H *= (ratio.T@W).T/(W.sum(axis=0)[:, None]+tiny)
        prediction = observed_predictions(W, H, rows, columns)
        ratio.data = counts/(prediction+tiny)
        W *= (ratio@H.T)/(H.sum(axis=1)[None, :]+tiny)
        prediction = observed_predictions(W, H, rows, columns)
        history[t] = fixed_term-np.sum(counts*np.log(prediction+tiny))+W.sum(axis=0)@H.sum(axis=1)
        if (t+1) % 10 == 0:
            print(f'NMF iteration {t+1}/100: divergence={history[t]:.6f}, '
                  f'{time.perf_counter()-started:.1f}s', flush=True)
    assert np.isfinite(history).all() and np.all(np.diff(history) <= 1e-6)
    assert np.all(W >= 0) and np.all(H >= 0)
    mass = W.sum(axis=0)
    W /= mass[None, :]
    H *= mass[:, None]
    np.testing.assert_allclose(W.sum(axis=0), 1, atol=1e-12)
    normalized_prediction = observed_predictions(W, H, rows, columns)
    np.testing.assert_allclose(normalized_prediction, prediction, rtol=1e-12, atol=1e-12)
    topics = []
    for k in range(25):
        top = np.argsort(-W[:, k], kind='stable')[:10]
        topics.append({'topic': k+1, 'top_words': [{'word': vocabulary[i],
                       'weight': float(W[i, k]), 'vocabulary_index': int(i+1)} for i in top]})
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.plot(np.arange(1, 101), history, linewidth=1.5)
    ax.set(xlabel='Iteration', ylabel='Generalized KL divergence D(X || WH)', xlim=(1, 100))
    ax.ticklabel_format(axis='y', style='plain', useOffset=False); ax.grid(alpha=.25)
    save_figure(fig, 'p2a_divergence.png')
    np.savetxt(OUT/'p2_objectives.csv', np.column_stack([np.arange(1, 101), history]),
               delimiter=',', header='iteration,divergence', comments='')
    np.savez_compressed(OUT/'p2_factors.npz', W=W, H=H)
    with (OUT/'p2_top_words.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['topic', 'rank', 'word', 'normalized_weight', 'vocabulary_index'])
        for topic in topics:
            for rank, word in enumerate(topic['top_words'], 1):
                writer.writerow([topic['topic'], rank, word['word'], word['weight'], word['vocabulary_index']])
    result = {'shape': list(X.shape), 'nonzeros': X.nnz, 'rank': 25, 'iterations': 100,
              'seed': 5720, 'initialization': 'Uniform(1,2)',
              'first_divergence': float(history[0]), 'final_divergence': float(history[-1]),
              'topics': topics}
    (OUT/'p2_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', choices=['all', '1', '2'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.problem in ('all', '1'):
        problem1()
    if args.problem in ('all', '2'):
        problem2()


if __name__ == '__main__':
    main()
