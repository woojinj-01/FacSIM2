import numpy as np

from facsimlib.processing import construct_network


def rank_diff_by_add1(net):

    ranks_wo_add = net.set_ranks(add_one=False).ranks(inverse=True)
    ranks_with_add = net.set_ranks(add_one=True).ranks(inverse=True)

    inst_list = list(ranks_wo_add.keys())

    rank_diff_dict = {}

    for inst in inst_list:

        rank_diff = np.abs(ranks_wo_add[inst] - ranks_with_add[inst])

        rank_diff_dict[inst] = rank_diff

    q3 = np.percentile(list(rank_diff_dict.values()), [75])

    for inst in inst_list:

        if rank_diff_dict[inst] > q3:
            print(inst)


if __name__ == "__main__":

    nets = construct_network(net_type='domestic')

    for net in nets.values():
        print("===", net.name)
        rank_diff_by_add1(net)



