import matplotlib.pyplot as plt
import numpy as np
import math

import facsimlib.processing
import facsimlib.math
from facsimlib.academia import Field
from facsimlib.text import get_country_code, normalize_inst_name


def plot_lorentz_curve_out_degree_integrated(network_list):
    
    network_names = [net.name for net in network_list]
    div_10per = [i for i in range(0, 101, 20)]

    # colors = ['#4169E1', '#2E8B57', '#C71585']
    colors = ['green', 'blue', 'red']
    color_ptr = 0

    fig_path = f"./fig/lc_integrated_{'_'.join(network_names)}.png"

    font = {'family': 'Helvetica Neue', 'size': 9}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=200)

    title = "Lorentz Curve on Out Degree"
    x_label = "Cumulative Ratio of Nodes (Unit: Percentage)"
    y_label = "Cumulative Ratio of Out Degree (Unit: Percentage)"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xlim(0, 100)
    plt.ylim(0, 100)

    for net in network_list:

        out_degrees = [net.net.out_degree(node) for node in net.net.nodes]

        (gini_coeff, x_co, y_co, base_co) = _process_gini_coeff(out_degrees)

        plt.plot(x_co, y_co, color=colors[color_ptr], linewidth=0.7, label=net.name, alpha=0.7)
        plt.scatter(div_10per, [_sample_from_data(x_co, y_co, index) for index in div_10per], \
                    c=colors[color_ptr], s=10, clip_on=False, alpha=1, marker='s')

        plt.plot(base_co, base_co, color='black', linewidth=0.7)

        color_ptr += 1

    plt.legend()
    plt.savefig(fig_path)
    plt.clf()


def plot_lorentz_curve_out_degree(network):

    fig_path = f"./fig/lc_{network.name}.png"

    out_degrees = [network.net.out_degree(node) for node in network.net.nodes]
    div_10per = [i for i in range(0, 101, 20)]
    
    (gini_coeff, x_co, y_co, base_co) = _process_gini_coeff(out_degrees)

    font = {'family': 'Helvetica Neue', 'size': 9}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=200)

    title = f"Lorentz Curve on Out Degree (Network: {network.name})"
    x_label = "Cumulative Ratio of Nodes (Unit: Percentage)"
    y_label = "Cumulative Ratio of Out Degree (Unit: Percentage)"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.plot(x_co, y_co, linewidth=0.7, alpha=0.7, c='blue')
    plt.scatter(div_10per, [_sample_from_data(x_co, y_co, index) for index in div_10per], \
                c='blue', s=10, clip_on=False, alpha=1, marker='s')
    plt.plot(base_co, base_co, color='black', linewidth=0.7)

    # plt.fill_between(x_co, y_co, base_co, alpha=0.2, color='grey')

    plt.savefig(fig_path)
    plt.clf()


def _process_gini_coeff(data):

    data.sort()

    total_num = len(data)
    total_sum = sum(data)
    percentage_delta = np.float32((1 / (total_num - 1) * 100))
    height_1 = np.float32(data[0] / total_sum * 100)
    height_2 = np.float32(data[0] + data[1] / total_sum * 100)

    area_AnB = (100 * 100) / 2
    area_B = 0

    x_co = []
    y_co = []
    base_co = []

    for i in range(total_num - 1):
        area_B += np.float32(percentage_delta * (height_1 + height_2) / 2)

        x_co.append(percentage_delta * i)
        y_co.append(height_1)
        base_co.append(percentage_delta * i)

        if (total_num - 2 != i):
            height_1 = height_2
            height_2 += np.float32(data[i + 2] / total_sum * 100)

    gini_coeff = np.float32((area_AnB - area_B) / area_AnB)
    
    x_co.append(np.float32(100))
    y_co.append(np.float32(100))
    base_co.append(np.float32(100))

    return (gini_coeff, x_co, y_co, base_co)


def _straight_line_equation(first_point, second_point):
    def wrapper(argXCo):

        x_co_first = first_point[0]
        y_co_first = first_point[1]

        x_co_second = second_point[0]
        y_co_second = second_point[1]

        gradient = (y_co_second - y_co_first) / (x_co_second - x_co_first)

        return gradient * (argXCo - x_co_first) + y_co_first
    return wrapper


def _sample_from_data(x_co, y_co, x_co_sample):

    if (any(not isinstance(targetList, list) for targetList in [x_co, y_co])):
        return None
    elif (sorted(x_co) != x_co):
        return None
    elif (len(x_co) != len(y_co)):
        return None
    elif (len(x_co) < 2):
        return None
    
    if (x_co_sample in x_co):
        return y_co[x_co.index(x_co_sample)]
    
    elif (min(x_co) > x_co_sample):

        low_x_ind = 0
        high_x_ind = low_x_ind + 1

    elif (max(x_co) < x_co_sample):

        low_x_ind = len(x_co) - 2
        high_x_ind = low_x_ind + 1

    else:
        for low_x_ind in range(len(x_co) - 1):

            high_x_ind = low_x_ind + 1

            if (x_co[low_x_ind] < x_co_sample < x_co[high_x_ind]):
                break

    def _straight_line_equation(first_point, second_point):
        def wrapper(argXCo):

            x_co_first = first_point[0]
            y_co_first = first_point[1]

            x_co_second = second_point[0]
            y_co_second = second_point[1]

            gradient = (y_co_second - y_co_first) / (x_co_second - x_co_first)

            return gradient * (argXCo - x_co_first) + y_co_first
        
        return wrapper

    lowXCo = x_co[low_x_ind]
    lowYCo = y_co[low_x_ind]

    highXCo = x_co[high_x_ind]
    highYCo = y_co[high_x_ind]

    line_eq = _straight_line_equation((lowXCo, lowYCo), (highXCo, highYCo))

    return line_eq(x_co_sample)


def plot_nonkr_bar(network: Field, group_size: int = 1, normalized: bool = False):  # clustering should be implemented

    if (not isinstance(network, Field)):
        return False
    elif (not isinstance(group_size, int) or group_size <= 0):
        return False
    elif (normalized not in [False, True]):
        return False

    str_list = []

    str_list.append(f"./fig/nonkr_{network.name}")
    
    if (group_size > 1):
        str_list.append(f"Group{group_size}")
    
    if (normalized is True):
        str_list.append("Norm")
    
    fig_path = "_".join(str_list) + ".png"

    total_count = {}
    kr_count = {}

    for node in network.net.nodes:

        edges_in = network.net.in_edges(node, data=True)

        rank = network.inst(node)['rank']

        if (rank is None):
            continue

        total_count[rank] = len(edges_in)
        kr_count[rank] = 0

        for _, _, data in edges_in:

            if (get_country_code(data['u_name']) == 'KR'):
                kr_count[rank] += 1

    nonkr_count = {rank: total_count[rank] - kr_count[rank] for rank in list(total_count.keys())}

    ranks_sorted = [key for key, _ in sorted(total_count.items())]
    kr_count_sorted = [value for _, value in sorted(kr_count.items())]
    nonkr_count_sorted = [value for _, value in sorted(nonkr_count.items())]

    if (group_size == 1):

        x_co = ranks_sorted
        y_co_kr = kr_count_sorted
        y_co_nonkr = nonkr_count_sorted

    else:
        
        x_co = []
        y_co_kr = []
        y_co_nonkr = []

        index = 0
        group_id = 1

        while index < len(ranks_sorted):

            elements_kr = []
            elements_nonkr = []

            while (group_size > len(elements_nonkr)):

                elements_kr.append(kr_count_sorted[index])
                elements_nonkr.append(nonkr_count_sorted[index])

                index += 1

                if (index >= len(ranks_sorted)):
                    break

            x_co.append(group_id)
            y_co_kr.append(sum(elements_kr))
            y_co_nonkr.append(sum(elements_nonkr))

            group_id += 1

    if (normalized is True):
            
        y_co_kr_norm = [y_co_kr[i] / (y_co_kr[i] + y_co_nonkr[i]) if (y_co_kr[i] + y_co_nonkr[i]) != 0 else 0 for i in range(len(x_co))]
        y_co_nonkr_norm = [y_co_nonkr[i] / (y_co_kr[i] + y_co_nonkr[i]) if (y_co_kr[i] + y_co_nonkr[i]) != 0 else 0 for i in range(len(x_co))]

        y_co_kr = y_co_kr_norm
        y_co_nonkr = y_co_nonkr_norm

    font = {'family': 'Helvetica Neue', 'size': 9}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=200)

    title = f"Doctorate Country for Assistant Professors (Network: {network.name})"
    x_label = "Number of Assistant Professors" if normalized is False else "Number of Assistant Professors (Normalized)"
    y_label = "Rank" if group_size == 1 else f"Group ({group_size} ranks/group)"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if normalized:
        plt.xlim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.25))

    plt.yticks(range(1, len(x_co) + 1))

    edge_width = 1 if 15 / len(x_co) > 1 else 15 / len(x_co)

    plt.barh(x_co, y_co_nonkr, color='blue', label='Earned doctorate abroad', alpha=0.7, edgecolor='black', linewidth=edge_width)
    plt.barh(x_co, y_co_kr, color='green', left=y_co_nonkr, label='Earned doctorate in S.Korea', alpha=0.7, edgecolor='black', linewidth=edge_width)
    
    plt.legend()

    plt.gca().invert_yaxis()

    plt.savefig(fig_path)

    plt.clf()


def plot_relative_rank_move(network: Field, percent_low=0, percent_high=100, normalized=False):

    if (not isinstance(network, Field)):
        return None
    
    if (any(int(num) != num for num in [percent_low, percent_high])):
        return None
    elif (not all(0 <= num <= 100 for num in [percent_low, percent_high])):
        return None
    elif (not percent_low < percent_high):
        return None
    
    rank_length = network.rank_length
    
    min_limit = 1 if percent_low == 0 else math.ceil(float(rank_length * percent_low / 100))
    max_limit = rank_length if percent_high == 100 else math.floor(float(rank_length * percent_high / 100))
    
    def rank_move(u_name, v_name, network: Field):

        u_name_norm = normalize_inst_name(u_name)
        v_name_norm = normalize_inst_name(v_name)

        ranks = network.ranks(inverse=True)

        u_rank = ranks[u_name_norm]
        v_rank = ranks[v_name_norm]

        if (all(rank is not None for rank in [u_rank, v_rank])):
            if (min_limit <= u_rank <= max_limit):
                return ranks[u_name_norm] - ranks[v_name_norm]
            
        return None

    if (percent_low == 0 and percent_high == 100):
        fig_path = f"./fig/rankmove_{network.name}.png" if normalized is False else f"./fig/rankmove_{network.name}_norm.png"

    else:
        fig_path = f"./fig/rankmove_{network.name}_{percent_low}_{percent_high}.png" if normalized is False \
            else f"./fig/rankmove_{network.name}_{percent_low}_{percent_high}_norm.png"

    rank_moves = []
    
    for move in list(network.net.edges):

        r_move = rank_move(move[0], move[1], network)

        if (r_move is not None):

            rank_moves.append(r_move / rank_length)

    bins = np.linspace(-1, 1, 40)

    hist, _ = np.histogram(rank_moves, bins=bins)

    font = {'family': 'Helvetica Neue', 'size': 9}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=200)

    title = f"Relative Rank Change Distribution (Network: {network.name})"
    x_label = "Relative Rank Change"
    y_label = "Number of Alumni" if normalized is False else "Number of Alumni (Normalized)"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xlim(-1, 1)

    if normalized is True:

        normalized_hist = hist / np.sum(hist)

        plt.ylim(0, max(normalized_hist) + 0.1 if max(normalized_hist) > 0.5 else 0.5)

        plt.bar([x + 0.025 for x in bins[:-1]], normalized_hist, width=np.diff(bins), alpha=0.7, color='blue', edgecolor='black')

    else:
        plt.bar([x + 0.025 for x in bins[:-1]], hist, width=np.diff(bins), alpha=0.7, color='blue', edgecolor='black')

    # plt.plot(bins_interp, spline(bins_interp), 'r-')

    # plt.legend()

    plt.savefig(fig_path)
    plt.clf()


def plot_up_down_hires(network_list, normalized: bool = False):

    if (not isinstance(normalized, bool)):
        return None

    network_names = [net.name for net in network_list]

    fig_path = f"./fig/hires_updown_{'_'.join(network_names)}.png" if normalized is False \
        else f"./fig/hires_updown_{'_'.join(network_names)}_Norm.png"

    up_hires = []
    self_hires = []
    down_hires = []

    for net in network_list:

        stat = facsimlib.math.up_down_hires(net, normalized=normalized)

        up_hires.append(stat[0])
        self_hires.append(stat[1])
        down_hires.append(stat[2])

    # colors = ['#4169E1', '#2E8B57', '#C71585']

    font = {'family': 'Helvetica Neue', 'size': 9}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=200)

    title = "Distribution of Hires"
    x_label = "Network"
    y_label = "Hires" if normalized is False else "Hires (Normalized)"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if normalized:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.25))

    plt.bar(network_names, down_hires, color='blue', label='Down hires', alpha=0.7, edgecolor='black')
    plt.bar(network_names, self_hires, color='red', bottom=down_hires, label='Self hires', alpha=0.7, edgecolor='black')
    plt.bar(network_names, up_hires, color='green', bottom=[down_hires[i] + self_hires[i] for i in range(len(down_hires))], \
            label='Up hires', alpha=0.7, edgecolor='black')

    plt.legend()

    plt.savefig(fig_path)
    
    plt.clf()


def plot_up_down_hires_zscore_random(network_src_list, trial=500):

    network_names = [net.name for net in network_src_list]

    fig_path = f"./fig/hires_updown_zscore_{'_'.join(network_names)}.png"

    z_dict = {}

    for net_src in network_src_list:

        net_src.set_ranks()

        up_hires_rand = []
        self_hires_rand = []
        down_hires_rand = []

        (up_src, self_src, down_src) = facsimlib.math.up_down_hires(net_src)

        net_src_rand = net_src.randomize(trial)

        for net_rand in net_src_rand:

            net_rand.set_ranks()

            (up, se, do) = facsimlib.math.up_down_hires(net_rand)

            up_hires_rand.append(up)
            self_hires_rand.append(se)
            down_hires_rand.append(do)

        up_z = (up_src - np.mean(up_hires_rand)) / np.std(up_hires_rand)
        self_z = (self_src - np.mean(self_hires_rand)) / np.std(self_hires_rand)
        down_z = (down_src - np.mean(down_hires_rand)) / np.std(down_hires_rand)

        z_dict[net_src.name] = (up_z, self_z, down_z)

    font = {'family': 'Helvetica Neue', 'size': 9}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=200)

    title = f"Z-Score of Hires (v.s. Random {trial} trials)"
    x_label = "Category"
    y_label = "Z-Score"

    # Set the width of the bars
    bar_width = 0.25

    # Adjust x positions for each set of bars
    x_positions1 = np.arange(len(network_names))
    x_positions2 = [x + bar_width for x in x_positions1]
    x_positions3 = [x + 2 * bar_width for x in x_positions1]

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xticks([x + bar_width for x in x_positions1], network_names)

    plt.bar(x_positions1, [z[0] for z in z_dict.values()], width=bar_width, label="Up hires", color='green', alpha=0.7, edgecolor='black')
    plt.bar(x_positions2, [z[1] for z in z_dict.values()], width=bar_width, label="Self hires", color='blue', alpha=0.7, edgecolor='black')
    plt.bar(x_positions3, [z[2] for z in z_dict.values()], width=bar_width, label="Down hires", color='red', alpha=0.7, edgecolor='black')

    plt.axhline(0, color='black', linewidth=1)

    plt.legend()

    plt.savefig(fig_path)
    plt.show()

    plt.clf()


def plot_rank_comparison(network_u: Field, network_v: Field, normalized=True):

    if not isinstance(network_u, Field) or not isinstance(network_v, Field):
        return None
    elif not isinstance(normalized, bool):
        return None

    if normalized:
        fig_path = f"./fig/rank_comparison_normalized_{network_u.name}_{network_v.name}.png"
    else:
        fig_path = f"./fig/rank_comparison_{network_u.name}_{network_v.name}.png"
    
    (rank_common_u, rank_common_v) = facsimlib.math._extract_common_ranks(network_u, network_v, normalized)

    max_rank_u = len(network_u.ranks()) if normalized is False else 1 
    max_rank_v = len(network_v.ranks()) if normalized is False else 1

    font = {'family': 'Helvetica Neue', 'size': 9}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=200)

    if normalized:

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.xticks(np.arange(0, 1.1, 0.25))
        plt.yticks(np.arange(0, 1.1, 0.25))

        title = "Rank Comparison"
        x_label = f"Normalized Rank (Network: {network_u.name})"
        y_label = f"Normalized Rank (Network: {network_v.name})"

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.scatter(rank_common_u, rank_common_v, s=20, marker='x', c='#4169E1')

        plt.plot([0, 1], [0, 1], c='black', linewidth=0.5)

        plt.savefig(fig_path)
        plt.clf()
    
    else:
        
        plt.xlim(1, max_rank_u)
        plt.ylim(1, max_rank_v)

        plt.xticks(range(1, max_rank_u + 1, math.floor(max_rank_u / 4)))
        plt.yticks(range(1, max_rank_v + 1, math.floor(max_rank_v / 4)))

        title = "Rank Comparison"
        x_label = f"Rank (Network: {network_u.name})"
        y_label = f"Rank (Network: {network_v.name})"

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.scatter(rank_common_u, rank_common_v, s=20, marker='x', c='#4169E1')

        plt.plot([1, max_rank_u], [1, max_rank_v], c='black', linewidth=0.5)

        plt.savefig(fig_path)
        plt.clf()


def plot_rank_comparison_multiple(network_u: Field, network_v_list: list):

    fig_path = f"./fig/rank_comparison_multiple_{network_u.name}.png"

    rank_common_v_list = []

    rank_common_v_mean = []
    rank_common_v_std = []

    for network_v in network_v_list:
    
        (rank_common_u, rank_common_v) = facsimlib.math._extract_common_ranks(network_u, network_v)

        rank_common_v_list.append(rank_common_v)

        max_rank_u = len(network_u.ranks())
        max_rank_v = len(network_v.ranks())

    for i in range(len(rank_common_u)):

        rank_gathered = [ranks[i] for ranks in rank_common_v_list]

        rank_common_v_mean.append(np.mean(rank_gathered))
        rank_common_v_std.append(np.max([np.max(rank_gathered) - np.mean(rank_gathered), np.mean(rank_gathered) - np.min(rank_gathered)]))

    font = {'family': 'Helvetica Neue', 'size': 9}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=200)

    plt.xlim(1, max_rank_u)
    plt.ylim(1, max_rank_v)

    plt.xticks(range(1, max_rank_u + 1, 50))
    plt.yticks(range(1, max_rank_v + 1, 50))

    title = "Rank Comparison"
    x_label = f"Rank (Network: {network_u.name})"
    y_label = "Rank (Network: Target Networks)"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.errorbar(rank_common_u, rank_common_v_mean, rank_common_v_std, \
                 linestyle='None', marker='s', mfc='cornflowerblue', mec='cornflowerblue', \
                 ms=1.5, capsize=2, ecolor='slategrey', elinewidth=0.5)

    plt.plot([1, max_rank_u], [1, max_rank_v], c='black', linewidth=0.5)

    plt.savefig(fig_path)
    plt.clf()


if (__name__ == "__main__"):

    network_dict = facsimlib.processing.construct_network()

    trial = 20

    plot_up_down_hires_zscore_random(list(network_dict.values()))

    net_closed = [net.closed.set_ranks() for net in network_dict.values()]

    plot_up_down_hires_zscore_random(net_closed)
        

    

