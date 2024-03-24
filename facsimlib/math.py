import scipy.stats
import math
import numpy as np
import networkx as nx

from facsimlib.academia import Field, NodeSelect as NS
from facsimlib.text import normalize_inst_name, area_seoul, area_capital, area_metro, area_others, inst_ists


def _extract_common_ranks(network1: Field, network2: Field, normalized=False):

    rank_list_1 = []
    rank_list_2 = []

    ranks_net1 = network1.ranks(inverse=True, normalized=normalized)
    ranks_net2 = network2.ranks(inverse=True, normalized=normalized)

    inst_names = sorted(list(network1.net.nodes))

    for name in inst_names:

        if (name not in ranks_net1 or name not in ranks_net2):
            continue

        rank1 = ranks_net1[name]
        rank2 = ranks_net2[name]

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


def up_down_hires(network: Field, normalized: bool = False):

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

    if normalized is False:
        return (up_hire, self_hire, down_hire)
    
    else:
        total_hire = up_hire + self_hire + down_hire

        return (up_hire / total_hire, self_hire / total_hire, down_hire / total_hire)


def up_down_hires_adv(network: Field, normalized: bool = False):

    def get_group_number(rank, rank_len, group_len):

        group_size = math.floor(rank_len / group_len)

        group_num = math.ceil(rank / group_size)

        if group_num > group_len:
            return group_len
        else:
            return group_num

    if (not isinstance(network, Field)):
        return None

    rank_len = network.rank_length
    group_len = 10

    ranks = network.ranks(inverse=True)
    hire_matrix = np.zeros((group_len, group_len))
    np.set_printoptions(linewidth=np.inf)
    
    for move in list(network.net.edges):

        r_move = rank_move(move[0], move[1], network)

        if (r_move is None):
            continue

        src_group = get_group_number(ranks[move[0]], rank_len, group_len)
        dst_group = get_group_number(ranks[move[1]], rank_len, group_len)

        hire_matrix[src_group - 1][dst_group - 1] += 1

    if normalized is False:
        return hire_matrix
    
    else:
        total_hire = np.sum(hire_matrix)
        
        return hire_matrix / total_hire


def paired_t_test_rank(network_u: Field, network_v: Field):

    ranks_dict_u = network_u.ranks(inverse=True)
    ranks_dict_v = network_v.ranks(inverse=True)
    
    ranks_u = [ranks_dict_u[key] for key in sorted(ranks_dict_u.keys())]
    ranks_v = [ranks_dict_v[key] for key in sorted(ranks_dict_v.keys())]

    return scipy.stats.ttest_rel(ranks_u, ranks_v)


def calc_sparsity(network: Field):

    adj_mat = nx.to_numpy_array(network.net)

    num_zeros = np.count_nonzero(adj_mat == 0)

    total_elem = adj_mat.size

    return num_zeros / total_elem


def trans_prob_between_regions(network: Field):

    adj_mat = nx.to_numpy_array(network.net)

    instlist = list(network.net.nodes())
    instcnt = 0

    labels = ["-IST", "Seoul", "Capital Area", "Metropolitan Cities", "Others"]

    ns_list = [NS('name', inst_ists, 'in', label="-IST"),
          NS('region', area_seoul, 'in', label="Seoul"),
          NS('region', area_capital, 'in', label="Capital Area"),
          NS('region', area_metro, 'in', label="Metropolitan\nCities"),
          NS('region', area_others, 'in', label="Others")]
    
    cols_for_label = {lab: [] for lab in labels}

    for inst in instlist:

        for label, ns in zip(labels, ns_list):

            if ns.hit(network.net.nodes[inst]):
                cols_for_label[label].append(instcnt)

                break

        instcnt += 1

    mat_compressed = np.zeros((len(adj_mat), len(labels)))

    for i in range(len(labels)):

        label = labels[i]
        cols = cols_for_label[label]

        for col in cols:
            mat_compressed[:, i] += adj_mat[:, col]
    
    mat_compressed_again = np.zeros((len(labels), len(labels)))

    for i in range(len(labels)):
        
        label = labels[i]
        cols = cols_for_label[label]

        for col in cols:
            mat_compressed_again[i, :] += mat_compressed[col, :]

    for i in range(len(labels)):

        row = mat_compressed_again[i, :]

        mat_compressed_again[i, :] = row / np.sum(row)

    return mat_compressed_again

        



