import numpy as np
import heapq
from scipy.sparse import csr_matrix

def discrete(X):
    for x in X.data:
        if x != 0 and x != 1:
            return False
    return True


def get_discrete_features(X, i, discrete=False):
    features = X.getcol(i).todense()
    if discrete:
        return np.array(features, dtype=int)
    mean = features.mean()
    return np.where(features >= mean, 1, 0).astype(int)


def calculate_conditional_entropy(Xi, y):
    size = y.shape[0]
    count = np.zeros([2, 2], dtype=int)
    for j in range(size):
        count[Xi[j][0]][y[j]] += 1
    p = count/size
    p0, p1 = p[0].sum(), p[1].sum()
    return f1(p[0][0], p0) + f1(p[0][1], p0) + f1(p[1][0], p1) + f1(p[1][1], p1)


def get_best_features(num_of_features, X, y):
    d = discrete(X)
    entropy_list = []
    entropy_records = dict()
    for i in range(X.shape[1]):
        e = calculate_conditional_entropy(get_discrete_features(X, i, d), y)
        entropy_records[e] = i
        heapq.heappush(entropy_list, e)
    best_entropy = heapq.nsmallest(num_of_features, entropy_list)
    best_feature_idx_list = [entropy_records[e] for e in best_entropy]
    return csr_matrix(np.hstack([X.getcol(i).todense() for i in best_feature_idx_list])), best_feature_idx_list


def f1(p, q):
    if p == 0:
        return 0.0
    return p * np.log(q/p)