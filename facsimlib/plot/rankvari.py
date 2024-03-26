import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
import numpy as np
# import scipy.stats

import facsimlib.processing
import facsimlib.math
from facsimlib.academia import NodeSelect as NS
from facsimlib.text import area_seoul, area_capital, area_metro, area_others
from facsimlib.plot.params import *


def rank_variation():

    fig_path = "./fig/rank_variation.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.05, 0.25), fontsize=param_tick_size)

    _rank_variation(network_dict["Biology"], ax[0])
    _rank_variation(network_dict["Computer Science"], ax[1])
    _rank_variation(network_dict["Physics"], ax[2])

    x_label = "Domestic Rank (Normalized)"
    y_label = "Domestic Rank - Global Rank (Normalized)"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(0, 1)
        axi.set_xticks(np.arange(0, 1.1, 0.25))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles = [Patch(facecolor=network_dict["Biology"].color, label="Biology", alpha=param_alpha,
                     edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ["Biology", "Computer Science", "Physics"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, frameon=False)
        
    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def rank_variation_random():

    fig_path = "./fig/rank_variation_random.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.05, 0.25), fontsize=param_tick_size)

    variations = _rank_variation(network_dict["Biology"], ax[0])
    _rank_variation_random(network_dict["Biology"], ax[0], to_compare=variations)

    variations = _rank_variation(network_dict["Computer Science"], ax[1])
    _rank_variation_random(network_dict["Computer Science"], ax[1], to_compare=variations)

    variations = _rank_variation(network_dict["Physics"], ax[2])
    _rank_variation_random(network_dict["Physics"], ax[2], to_compare=variations)

    x_label = "Domestic Rank (Normalized)"
    y_label = "Domestic Rank - Global Rank (Normalized)"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(0, 1)
        axi.set_xticks(np.arange(0, 1.1, 0.25))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles = [Patch(facecolor=network_dict["Biology"].color, label="Biology",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Line2D([], [], marker='o', color='black', markerfacecolor='black', markersize=10, linestyle='None'),
               Line2D([], [], marker='x', color='black', markerfacecolor='black', markersize=10, linestyle='None')]

    labels = ["Biology", "Computer Science", "Physics", "Our data", "Randomized data"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, frameon=False)
        
    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def rank_variation_random_zscore():

    fig_path = "./fig/rank_variation_random_zscore.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    # plt.ylim(-5, 5)
    # plt.yticks(np.arange(-5, 6, 2.5), fontsize=param_tick_size)

    variations = _rank_variation(network_dict["Biology"], None)
    _rank_variation_random_zscore(network_dict["Biology"], ax[0], to_compare=variations, marker='o')

    variations = _rank_variation(network_dict["Computer Science"], None)
    _rank_variation_random_zscore(network_dict["Computer Science"], ax[1], to_compare=variations, marker='o')

    variations = _rank_variation(network_dict["Physics"], None)
    _rank_variation_random_zscore(network_dict["Physics"], ax[2], to_compare=variations, marker='o')

    x_label = "Domestic Rank (Normalized)"
    y_label = "Z-score"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(0, 1)
        axi.set_xticks(np.arange(0, 1.1, 0.25))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles = [Patch(facecolor=network_dict["Biology"].color, label="Biology", alpha=param_alpha,
                     edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ["Biology", "Computer Science", "Physics"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False)
        
    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def rank_variation_random_zscore_vs_ratio():

    fig_path = "./fig/rank_variation_random_zscore_vs_ratio.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    # plt.ylim(-5, 5)
    # plt.yticks(np.arange(-5, 6, 2.5), fontsize=param_tick_size)

    variations = _rank_variation(network_dict["Biology"], None)
    _rank_variation_random_zscore_vs_ratio(network_dict["Biology"], ax[0], to_compare=variations, marker='o')

    variations = _rank_variation(network_dict["Computer Science"], None)
    _rank_variation_random_zscore_vs_ratio(network_dict["Computer Science"], ax[1], to_compare=variations, marker='o')

    variations = _rank_variation(network_dict["Physics"], None)
    _rank_variation_random_zscore_vs_ratio(network_dict["Physics"], ax[2], to_compare=variations, marker='o')

    x_label = "Ratio"
    y_label = "Z-score"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(0, 1)
        axi.set_xticks(np.arange(0, 1.1, 0.25))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles = [Patch(facecolor=network_dict["Biology"].color, label="Biology", alpha=param_alpha,
                     edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ["Biology", "Computer Science", "Physics"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False)
        
    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def rank_variation_random_zscore_mag_vs_ratio():

    fig_path = "./fig/rank_variation_random_zscore_mag_vs_ratio.pdf"

    cmap = 'jet_r'

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    zscores = []

    variations = _rank_variation(network_dict["Biology"], None)
    zscores += _rank_variation_random_zscore_mag_vs_ratio(network_dict["Biology"], ax[0], cmap, to_compare=variations, marker='o')

    variations = _rank_variation(network_dict["Computer Science"], None)
    zscores += _rank_variation_random_zscore_mag_vs_ratio(network_dict["Computer Science"], ax[1], cmap, to_compare=variations, marker='o')

    variations = _rank_variation(network_dict["Physics"], None)
    zscores += _rank_variation_random_zscore_mag_vs_ratio(network_dict["Physics"], ax[2], cmap, to_compare=variations, marker='o')

    x_label = "Ratio of Foreign Doctorates"
    y_label = "Number of Foreign Doctorates"

    fig.supxlabel(x_label, fontsize=param_xlabel_size)
    fig.supylabel(y_label, fontsize=param_ylabel_size)

    vmax = max(zscores)
    vmin = min(zscores)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    colormapping = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    cbar = fig.colorbar(colormapping, ax=plt.gca())

    cbar.set_label("Z-score", fontsize=param_ylabel_size * 0.7)

    for axi in ax.flatten():

        axi.set_xlim(0, 1)
        axi.set_xticks(np.arange(0, 1.1, 0.25))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)
        
    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _rank_variation(network, ax, marker='o'):

    if ax is None:
        do_plot = False
    else:
        do_plot = True

    network_c = network.domestic

    network.set_ranks()
    network_c.set_ranks()

    ns_list = [NS('region', area_seoul, 'in', label="Seoul"),
               NS('region', area_capital, 'in', label="Capital Area"),
               NS('region', area_metro, 'in', label="Metropolitan Cities"),
               NS('region', area_others, 'in', label="Others")]

    if do_plot:

        ax.set_xlim(0, 1)
        ax.hlines(0, -1, 1, colors='black', linestyles='dashed', linewidth=1)

        ax.set_xticks(np.arange(0, 1.1, 0.25))
        ax.tick_params(axis='both', which='both', labelsize=param_tick_size)

    to_compare = {}

    for ns in ns_list:

        net = network.filter(ns)
        net_c = network_c.filter(ns)

        (rank_common_u, rank_common_v) = facsimlib.math._extract_common_ranks(net, net_c, normalized=True)

        rank_variation = [rank_common_v[i] - rank_common_u[i] for i in range(len(rank_common_u))]

        if do_plot:
            ax.scatter(rank_common_v, rank_variation, s=70, marker=marker, c=network.color, alpha=param_alpha)

        to_compare_local = {rank_common_v[i]: rank_variation[i] for i in range(len(rank_variation))}

        to_compare.update(to_compare_local)

    return to_compare


def _rank_variation_random(network, ax, marker='x', to_compare=None):

    trial = 500

    def _get_rank_variation(network):

        network_c = network.domestic

        network.set_ranks()
        network_c.set_ranks()

        return facsimlib.math._extract_common_ranks(network, network_c, normalized=True), network_c.ranks(normalized=True)
    
    randomized = network.randomize(trial)

    vari_dict = {}

    for net_r in randomized:

        (ranks_u, ranks_v), dom_ranks = _get_rank_variation(net_r)

        for r_u, r_v in zip(ranks_u, ranks_v):

            vari = r_v - r_u

            if r_v not in vari_dict:
                vari_dict[r_v] = [vari]
            else:
                vari_dict[r_v].append(vari)

    for key, value in vari_dict.items():
        vari_dict[key] = sum(value) / len(value)

    if to_compare is not None:
        for r_v in vari_dict.keys():
            if r_v in to_compare and to_compare[r_v] < vari_dict[r_v]:
                print(f"{network.name}: {dom_ranks[r_v]} ({to_compare[r_v]}. {vari_dict[r_v]})")

    ax.set_xlim(0, 1)
    ax.hlines(0, -1, 1, colors='black', linestyles='dashed', linewidth=1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.tick_params(axis='both', which='both', labelsize=param_tick_size)

    ax.scatter(list(vari_dict.keys()), list(vari_dict.values()), s=70, marker=marker, c=network.color, alpha=param_alpha)


def _rank_variation_random_zscore(network, ax, marker='x', to_compare=None):

    trial = 500
    domestic_ranks = network.domestic.set_ranks().ranks(normalized=True)

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")

    def _get_rank_variation(network):

        network_c = network.domestic

        network.set_ranks()
        network_c.set_ranks()

        return facsimlib.math._extract_common_ranks(network, network_c, normalized=True), network_c.ranks(normalized=True)
    
    def _get_num_foreign_doc(rank):

        inst_name = domestic_ranks[rank]

        kr_count = 0
        nonkr_count = 0

        edges_in = network.net.in_edges(inst_name, data=True)

        for src, dst, data in edges_in:

            if (ns_korea.hit(network.net.nodes[src])):
                kr_count += 1

            else:
                nonkr_count += 1

        return nonkr_count
    
    randomized = network.randomize(trial)

    vari_dict = {}

    for net_r in randomized:

        (ranks_u, ranks_v), dom_ranks = _get_rank_variation(net_r)

        for r_u, r_v in zip(ranks_u, ranks_v):

            vari = r_v - r_u

            if r_v not in vari_dict:
                vari_dict[r_v] = [vari]
            else:
                vari_dict[r_v].append(vari)

    to_delete = []

    num_foreign = []

    stds = []

    for key, value in vari_dict.items():

        if key in to_compare:

            emp_data = to_compare[key]
            rand_mean = sum(value) / len(value)
            rand_std = np.std(value)

            z_score = (emp_data - rand_mean) / rand_std

            vari_dict[key] = z_score

            if z_score > 5:
                print(f"{network.name}: {domestic_ranks[key]}")

            num_foreign.append(_get_num_foreign_doc(key))
            stds.append(rand_std)

        else:
            to_delete.append(key)

    for key in to_delete:
        del vari_dict[key]

    x_co = list(vari_dict.keys())
    y_co = list(vari_dict.values())

    ax.set_xlim(0, 1)
    ax.hlines(0, -1, 1, colors='black', linestyles='dashed', linewidth=1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.tick_params(axis='both', which='both', labelsize=param_tick_size)

    for i in range(len(x_co)):

        scale = (num_foreign[i] + 1) / (max(num_foreign) + 1)

        ax.scatter(x_co[i], y_co[i], s=400 * scale, marker=marker, c=network.color, alpha=param_alpha)

    # sorted_data = sorted(zip(x_co, stds))
    # x_co_sorted, stds_sorted = zip(*sorted_data)

    # ax.plot(x_co_sorted, stds_sorted, marker='s', c='red', linestyle='--', alpha=param_alpha)


def _rank_variation_random_zscore_vs_ratio(network, ax, marker='x', to_compare=None):

    trial = 500
    domestic_ranks = network.domestic.set_ranks().ranks(normalized=True)

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")

    def _get_rank_variation(network):

        network_c = network.domestic

        network.set_ranks()
        network_c.set_ranks()

        return facsimlib.math._extract_common_ranks(network, network_c, normalized=True), network_c.ranks(normalized=True)
    
    def _get_ratio_foreign_doc(rank):

        inst_name = domestic_ranks[rank]

        kr_count = 0
        nonkr_count = 0

        edges_in = network.net.in_edges(inst_name, data=True)

        for src, dst, data in edges_in:

            if (ns_korea.hit(network.net.nodes[src])):
                kr_count += 1

            else:
                nonkr_count += 1

        total_count = kr_count + nonkr_count

        if total_count == 0:
            return None
        else:
            return nonkr_count / total_count
    
    randomized = network.randomize(trial)

    vari_dict = {}

    for net_r in randomized:

        (ranks_u, ranks_v), dom_ranks = _get_rank_variation(net_r)

        for r_u, r_v in zip(ranks_u, ranks_v):

            vari = r_v - r_u

            if r_v not in vari_dict:
                vari_dict[r_v] = [vari]
            else:
                vari_dict[r_v].append(vari)

    to_delete = []

    z_scores = []
    ratio_foreign = []
    
    for key, value in vari_dict.items():

        if key in to_compare:

            ratio = _get_ratio_foreign_doc(key) 

            if ratio is None:

                to_delete.append(key)
                continue

            emp_data = to_compare[key]
            rand_mean = sum(value) / len(value)
            rand_std = np.std(value)

            z_score = (emp_data - rand_mean) / rand_std

            if z_score > 5:
                print(f"Large Z: {network.name}: {domestic_ranks[key]}")
                
                to_delete.append(key)
                continue

            z_scores.append(z_score)
            ratio_foreign.append(ratio)

        else:
            to_delete.append(key)

    for key in to_delete:
        del vari_dict[key]

    # coeff = np.polyfit(ratio_foreign, z_scores, 1)

    # poly1d_func = np.poly1d(coeff)

    ax.hlines(0, -1, 1, colors='black', linestyles='dashed', linewidth=1)

    ax.tick_params(axis='both', which='both', labelsize=param_tick_size)

    ax.scatter(ratio_foreign, z_scores, s=200, marker=marker, c=network.color, alpha=param_alpha)

    # print(f"=== Correlations: {network.name}")
    # print(scipy.stats.pearsonr(ratio_foreign, z_scores))
    # print(scipy.stats.spearmanr(ratio_foreign, z_scores))
    # print(scipy.stats.kendalltau(ratio_foreign, z_scores))
    # print("\n")


def _rank_variation_random_zscore_mag_vs_ratio(network, ax, cmap, marker='x', to_compare=None):

    trial = 500
    domestic_ranks = network.domestic.set_ranks().ranks(normalized=True)

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")

    def _get_rank_variation(network):

        network_c = network.domestic

        network.set_ranks()
        network_c.set_ranks()

        return facsimlib.math._extract_common_ranks(network, network_c, normalized=True), network_c.ranks(normalized=True)
    
    def _get_num_and_ratio_foreign_doc(rank):

        inst_name = domestic_ranks[rank]

        kr_count = 0
        nonkr_count = 0

        edges_in = network.net.in_edges(inst_name, data=True)

        for src, dst, data in edges_in:

            if (ns_korea.hit(network.net.nodes[src])):
                kr_count += 1

            else:
                nonkr_count += 1

        total_count = kr_count + nonkr_count

        if total_count == 0:
            return None
        else:
            return (nonkr_count, nonkr_count / total_count)
    
    randomized = network.randomize(trial)

    vari_dict = {}

    for net_r in randomized:

        (ranks_u, ranks_v), dom_ranks = _get_rank_variation(net_r)

        for r_u, r_v in zip(ranks_u, ranks_v):

            vari = r_v - r_u

            if r_v not in vari_dict:
                vari_dict[r_v] = [vari]
            else:
                vari_dict[r_v].append(vari)

    to_delete = []

    z_scores = []
    ratio_foreign = []
    num_foreign = []
    
    for key, value in vari_dict.items():

        if key in to_compare:

            num_n_ratio = _get_num_and_ratio_foreign_doc(key) 

            if num_n_ratio is None:

                to_delete.append(key)
                continue

            emp_data = to_compare[key]
            rand_mean = sum(value) / len(value)
            rand_std = np.std(value)

            z_score = (emp_data - rand_mean) / rand_std

            if z_score > 5:
                print(f"{network.name}: {domestic_ranks[key]}")
                
                to_delete.append(key)
                continue

            num_foreign.append(num_n_ratio[0])
            ratio_foreign.append(num_n_ratio[1])
            z_scores.append(z_score)

        else:
            to_delete.append(key)

    for key in to_delete:
        del vari_dict[key]

    ax.tick_params(axis='both', which='both', labelsize=param_tick_size)

    ax.scatter(ratio_foreign, num_foreign, s=400, c=z_scores, marker=marker, edgecolor='black', cmap=cmap)

    return z_scores


if __name__ == "__main__":
    rank_variation_random_zscore()