import scipy.stats
import math
import numpy as np
from sklearn.cluster import KMeans

from facsimlib.academia import Field
from facsimlib.text import normalize_inst_name
from facsimlib.processing import construct_network


def _extract_common_ranks(network1: Field, network2: Field):

    rank_list_1 = []
    rank_list_2 = []

    inst_names = sorted(list(network1.net.nodes))

    for name in inst_names:

        if (any(inst is None for inst in [network1.inst(name), network2.inst(name)])):
            continue

        rank1 = network1.inst(name)['rank']
        rank2 = network2.inst(name)['rank']

        if (any(rank is None for rank in [rank1, rank2])):
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


def get_random_rank_tier(network: Field, trial=500):

    inst_names = sorted(list(network.net.nodes))

    target_to_compare = network.randomize(trial)

    rank_common_list = []

    rank_common_mean = []
    rank_common_std = []

    network.set_ranks()

    for net_rand in target_to_compare:

        net_rand.set_ranks()

        rank_common = _extract_common_ranks(network, net_rand)[1]

        rank_common_list.append(rank_common)

    for i in range(len(rank_common_list[0])):

        rank_gathered = [ranks[i] for ranks in rank_common_list]

        rank_common_mean.append(np.mean(rank_gathered))
        rank_common_std.append(np.std(rank_gathered))

    stat = {inst_names[i]: (rank_common_mean[i], rank_common_std[i]) for i in range(len(inst_names))}

    data = [stat[name][0] for name in inst_names]

    data_array = np.array(list(data)).reshape(-1, 1)

    # Specify the number of clusters you want to identify
    num_clusters = 4

    # Fit the KMeans model to your data
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data_array)

    # Get the cluster labels and centroids
    cluster_labels = kmeans.labels_

    # Create a dictionary to store clusters and their corresponding institutions
    clusters = {f'Cluster {i+1}': {'institutions': [], 'mean': 0} for i in range(num_clusters)}

    # Populate the clusters dictionary
    for institution, label in zip(inst_names, cluster_labels):
        clusters[f'Cluster {label+1}']['institutions'].append(institution)

    # Calculate the mean for each cluster
    for cluster, data in clusters.items():
        mean_value = np.mean([stat[institution][0] for institution in data['institutions']])
        clusters[cluster]['mean'] = mean_value

    # Order clusters by mean value
    ordered_clusters = dict(sorted(clusters.items(), key=lambda item: item[1]['mean']))

    # Print the ordered clusters

    tier = 1

    result = {}

    for _, data in ordered_clusters.items():

        result[f"Tier {tier}"] = (data['institutions'], data['mean'])

        tier += 1

    return result


if (__name__ == "__main__"):

    network_dict = construct_network()

    tier_1 = {}
    tier_2 = {}
    tier_3 = {}
    tier_4 = {}

    for net in network_dict.values():

        print("=" * 10, net.name, "=" * 10)

        tiers = get_random_rank_tier(net.closed)

        print(tiers)

        tier_1[net.name] = tiers["Tier 1"][0]
        tier_2[net.name] = tiers["Tier 2"][0]
        tier_3[net.name] = tiers["Tier 3"][0]
        tier_4[net.name] = tiers["Tier 4"][0]

    tier_1_common = set(tier_1["Biology"]) & set(tier_1["Computer Science"]) & set(tier_1["Physics"])
    tier_2_common = set(tier_2["Biology"]) & set(tier_2["Computer Science"]) & set(tier_2["Physics"])
    tier_3_common = set(tier_3["Biology"]) & set(tier_3["Computer Science"]) & set(tier_3["Physics"])
    tier_4_common = set(tier_4["Biology"]) & set(tier_4["Computer Science"]) & set(tier_4["Physics"])

    print(tier_1_common)
    print(tier_2_common)
    print(tier_3_common)
    print(tier_4_common)

    print(len(tier_1_common) / np.mean([len(tier_1[field]) for field in network_dict.keys()]))
    print(len(tier_2_common) / np.mean([len(tier_2[field]) for field in network_dict.keys()]))
    print(len(tier_3_common) / np.mean([len(tier_3[field]) for field in network_dict.keys()]))
    print(len(tier_4_common) / np.mean([len(tier_4[field]) for field in network_dict.keys()]))

    # cs = network_dict["Computer Science"].closed
    # cs_rand = cs.random
    
    # cs.set_ranks()
    # cs_rand.set_ranks()

    # print(_extract_common_ranks(cs, cs_rand))
    
