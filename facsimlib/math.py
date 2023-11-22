import scipy.stats
import math

from facsimlib.academia import Field
from facsimlib.text import normalize_inst_name


def _extract_common_ranks(network1: Field, network2: Field, normalized=False):

    rank_list_1 = []
    rank_list_2 = []

    ranks_net1 = network1.ranks(normalized=normalized)
    ranks_net2 = network2.ranks(normalized=normalized)

    inst_names = sorted(list(network1.net.nodes))

    for name in inst_names:

        if (name not in ranks_net1 or name not in ranks_net2):
            continue

        rank1 = ranks_net1[name]
        rank2 = ranks_net1[name]

        if (rank1 is None or rank2 is None):
            continue
        
        rank_list_1.append(rank1)
        rank_list_2.append(rank2)

    return (rank_list_1, rank_list_2)


def spearman_rank_correlation(network1: Field, network2: Field):

    (rank_list_1, rank_list_2) = _extract_common_ranks(network1, network2)

    spearman_corr, _ = scipy.stats.spearmanr(rank_list_1, rank_list_2)

    return round(spearman_corr, 4)


def pearson_rank_correlation(network1: Field, network2: Field):

    (rank_list_1, rank_list_2) = _extract_common_ranks(network1, network2)

    pearson_corr, _ = scipy.stats.pearsonr(rank_list_1, rank_list_2)

    return round(pearson_corr, 4)


def average_rank_difference(network1: Field, network2: Field, rms=False):

    if (not isinstance(rms, bool)):
        return None

    (rank_list_1, rank_list_2) = _extract_common_ranks(network1, network2)

    rank_difference = [rank_list_2[i] - rank_list_1[i] for i in range(len(rank_list_1))]

    if (len(rank_difference) == 0):
        return None
    
    else:

        if (rms is True):

            diff_squared = [x**2 for x in rank_difference]

            return math.sqrt(sum(diff_squared) / len(diff_squared))
        
        else:
            return sum(rank_difference) / len(rank_difference)
        
        
def rank_move(u_name, v_name, network: Field):

    u_name_norm = normalize_inst_name(u_name)
    v_name_norm = normalize_inst_name(v_name)

    ranks = network.ranks(inverse=True)

    u_rank = ranks[u_name_norm]
    v_rank = ranks[v_name_norm]

    if (all(rank is not None for rank in [u_rank, v_rank])):
        return ranks[u_name_norm] - ranks[v_name_norm]
    
    else:
        return None
        

def average_relative_rank_move(network: Field):

    if (not isinstance(network, Field)):
        return None
    
    rank_moves = []
    max_rank = 0

    for rank in network.ranks(inverse=True).values():

        if (rank is not None):
            max_rank += 1
    
    for move in list(network.net.edges):

        r_move = rank_move(move[0], move[1], network)

        if (r_move is not None):

            rank_moves.append(r_move / max_rank)

    return sum(rank_moves) / len(rank_moves)


def up_down_hires(network: Field):

    if (not isinstance(network, Field)):
        return None
    
    up_hire = 0
    self_hire = 0
    down_hire = 0
    
    for move in list(network.net.edges):

        r_move = rank_move(move[0], move[1], network)

        if (r_move is None):
            continue

        if (move[0] == move[1]):
            
            self_hire += 1

        elif (r_move > 0):
            up_hire += 1

        elif (r_move < 0):
            down_hire += 1

        else:
            pass

    return (up_hire, self_hire, down_hire)


if (__name__ == "__main__"):

    pass