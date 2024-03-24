import numpy as np
from facsimlib import SpringRank as sp


global_list = [[14, 2, 6, 9, 0, 0],
               [1, 8, 5, 6, 0, 0],
               [0, 0, 4, 1, 0, 0],
               [0, 0, 0, 3, 0, 0],
               [3, 2, 0, 0, 0, 0],
               [2, 2, 0, 0, 0, 0]]

domestic_list = [[14, 2, 6, 9],
                 [1, 8, 5, 6],
                 [0, 0, 4, 1],
                 [0, 0, 0, 3]]

domestic_arr = np.array(domestic_list)
global_arr = np.array(global_list)


def calc_sprank(mat):

    scores = sp.get_ranks(mat)

    sorted_indices = np.argsort(scores)[::-1]

    value_to_rank = {value: rank + 1 for rank, value in enumerate(sorted_indices)}

    integers_based_on_rank = [value_to_rank[value] for value in range(len(scores))]

    return integers_based_on_rank


domestic_ranks = calc_sprank(domestic_arr)
global_ranks = calc_sprank(global_arr)

print(domestic_ranks)
print(global_ranks)




