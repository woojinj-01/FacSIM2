import numpy as np
import math
from random import choices
from facsimlib import SpringRank as sp


def _solve_quad_eq(a, b, c):

    x1 = ((-b + (math.sqrt(math.pow(b, 2) - 4 * a * c))) / 2 * a)
    x2 = ((-b - (math.sqrt(math.pow(b, 2) - 4 * a * c))) / 2 * a)

    return x1, x2


def calc_sprank(mat):

    scores = sp.get_ranks(mat)

    sorted_indices = np.argsort(scores)[::-1]

    value_to_rank = {value: rank + 1 for rank, value in enumerate(sorted_indices)}

    integers_based_on_rank = [value_to_rank[value] for value in range(len(scores))]

    return integers_based_on_rank


def get_total_out_deg(len, i):

    return (len - i) * 5


def get_out_deg(deg_left):

    ratio = [0.7, 0.3, 0]
    prob = [0.5, 0.3, 0.2]

    sample = choices(ratio, prob)[0]

    return math.floor(deg_left * sample)


def gen_adjmat(len, gini, sparsity):

    assert (isinstance(len, int) and len >= 1)
    assert (isinstance(sparsity, float) and 0 <= sparsity <= 1)

    mat = np.zeros((len, len))

    # num_sparse_nodes = math.floor(len * sparsity)

    x1, x2 = _solve_quad_eq(1, -2 * len, sparsity * (len ** 2))

    if 0 < math.floor(x1) <= len:
        num_sparse_nodes = math.floor(x1)
    
    elif 0 < math.floor(x2) <= len:
        num_sparse_nodes = math.floor(x2)

    else:
        return None

    print(num_sparse_nodes)

    num_nonsparse_nodes = len - num_sparse_nodes - 1

    for i in range(0, num_nonsparse_nodes):

        out_deg = get_total_out_deg(len, i)

        deg_j = get_out_deg(out_deg)

        mat[i, i] = deg_j

        out_deg -= deg_j

        for j in range(0, num_nonsparse_nodes):

            if i == j:
                continue

            if out_deg <= 0:
                break

            deg_j = get_out_deg(out_deg)

            mat[i, j] = deg_j

            out_deg -= deg_j

    return mat


if __name__ == "__main__":

    mat = gen_adjmat(200, 0.1, 0.9)

    ranks = calc_sprank(mat)

    print(mat)
    print(ranks)    

    mat += 1

    ranks = calc_sprank(mat)
    print(ranks)

    