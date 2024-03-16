import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import numpy as np
import math
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import facsimlib.processing
import facsimlib.math
from facsimlib.academia import Field, NodeSelect as NS
from facsimlib.text import normalize_inst_name, area_seoul, area_capital, area_metro, area_others, inst_ists
from facsimlib.plot.params import *


def _interhistogram(x, y):

    f = interp1d(x, y, kind='quadratic', fill_value='extrapolate')

    x_co = np.arange(-1, 1.01, 0.01)
    y_co = f(x_co)

    return x_co, y_co


def figure_rank_move():

    fig_path = "./fig/rank_move.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(0, 0.25)
    plt.yticks(np.arange(0, 0.26, 0.125), fontsize=param_tick_size)

    _figure_rank_move(network_dict["Biology"], ax[0])
    _figure_rank_move(network_dict["Computer Science"], ax[1])
    _figure_rank_move(network_dict["Physics"], ax[2])

    x_label = "Relative Movement of Faculty (Normalized)"
    y_label = "Density"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=5),
               Line2D([0], [0], color='black', linestyle=':', linewidth=5),
               Patch(facecolor=network_dict["Biology"].color, label="Biology",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ["Global", "Domestic", "Biology", "Computer Science", "Physics"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=5, frameon=False)

    for axi in ax.flatten():

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    # ax[1].legend(fontsize=param_legend_size, loc='upper center', bbox_to_anchor=(0.5, 1.08),
    #              ncol=3, frameon=False, fancybox=True, shadow=True, handlelength=2.5)
    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _figure_rank_move(network, ax):

    network_c = network.domestic

    rank_moves = {}

    rank_moves[network.name] = ([], '-')
    rank_moves[network_c.name] = ([], ':')

    nets = [network, network_c]

    rank_length = network.rank_length
    
    def rank_move(u_name, v_name, network: Field):

        u_name_norm = normalize_inst_name(u_name)
        v_name_norm = normalize_inst_name(v_name)

        ranks = network.ranks(inverse=True)

        u_rank = ranks[u_name_norm]
        v_rank = ranks[v_name_norm]

        if (all(rank is not None for rank in [u_rank, v_rank])):
            return ranks[u_name_norm] - ranks[v_name_norm]
            
        return None
    
    bins = np.linspace(-1, 1, 40)
    
    for netw in nets:

        for move in list(netw.net.edges):

            r_move = rank_move(move[0], move[1], netw)

            if (r_move is not None):

                rank_moves[netw.name][0].append(r_move / rank_length)

    x = (bins[:-1] + bins[1:]) / 2

    for key, value in rank_moves.items():

        name = key
        (r_moves, style) = value

        if name == network.name:
            num_edge_ratio = len(r_moves) / network.net.number_of_edges()

        elif name == network_c.name:
            num_edge_ratio = len(r_moves) / network_c.net.number_of_edges()

        hist, _ = np.histogram(r_moves, bins=bins)
        normalized_hist = hist / np.sum(hist)

        (x_interp, y_interp) = _interhistogram(x, normalized_hist)

        y_interp = [elem * num_edge_ratio for elem in y_interp]

        ax.plot(x_interp, y_interp, color=network.color, linewidth=5, alpha=param_alpha, linestyle=style)


def figure_rank_move_3group(net_type='global'):

    fig_path = f"./fig/rank_move_3group_{net_type}.pdf"

    network_dict = facsimlib.processing.construct_network(net_type=net_type)

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(0, 0.25)
    plt.yticks(np.arange(0, 0.26, 0.125), fontsize=param_tick_size)

    _figure_rank_move_3group(network_dict["Biology"], ax[0])
    _figure_rank_move_3group(network_dict["Computer Science"], ax[1])
    _figure_rank_move_3group(network_dict["Physics"], ax[2])

    x_label = "Relative Movement of Faculty (Normalized)"
    y_label = "Density"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=5),
               Line2D([0], [0], color='black', linestyle=':', linewidth=5),
               Line2D([0], [0], color='black', linestyle='-.', linewidth=5),
               Patch(facecolor=network_dict["Biology"].color, label="Biology", alpha=param_alpha,
                     edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ["0-20%", "40-60%", "80-100%",
              network_dict["Biology"].name, network_dict["Computer Science"].name, network_dict["Physics"].name]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def figure_rank_move_3group_with_inset(net_type='global'):

    fig_path = f"./fig/rank_move_3group_{net_type}_inset.pdf"

    network_dict = facsimlib.processing.construct_network(net_type=net_type)

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(0, 0.25)
    plt.yticks(np.arange(0, 0.26, 0.125), fontsize=param_tick_size)

    _figure_rank_move_3group(network_dict["Biology"], ax[0], inset=True)
    _figure_rank_move_3group(network_dict["Computer Science"], ax[1], inset=True)
    _figure_rank_move_3group(network_dict["Physics"], ax[2], inset=True)

    x_label = "Relative Movement of Faculty (Normalized)"
    y_label = "Density"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=5),
               Line2D([0], [0], color='black', linestyle=':', linewidth=5),
               Line2D([0], [0], color='black', linestyle='-.', linewidth=5),
               Patch(facecolor=network_dict["Biology"].color, label="Biology", alpha=param_alpha,
                     edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ["0-20%", "40-60%", "80-100%",
              network_dict["Biology"].name, network_dict["Computer Science"].name, network_dict["Physics"].name]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def figure_rank_move_3group_separate(net_type='global'):

    fig_path = f"./fig/rank_move_3group_{net_type}_sep.pdf"

    network_dict = facsimlib.processing.construct_network(net_type=net_type)

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0, 0.51, 0.25), fontsize=param_tick_size)

    _figure_rank_move_3group(network_dict["Biology"], ax[0], normalize_sep=True)
    _figure_rank_move_3group(network_dict["Computer Science"], ax[1], normalize_sep=True)
    _figure_rank_move_3group(network_dict["Physics"], ax[2], normalize_sep=True)

    x_label = "Relative Movement of Faculty (Normalized)"
    y_label = "Density"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=5),
               Line2D([0], [0], color='black', linestyle=':', linewidth=5),
               Line2D([0], [0], color='black', linestyle='-.', linewidth=5),
               Patch(facecolor=network_dict["Biology"].color, label="Biology", alpha=param_alpha,
                     edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ["0-20%", "40-60%", "80-100%",
              network_dict["Biology"].name, network_dict["Computer Science"].name, network_dict["Physics"].name]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _figure_rank_move_3group(network, ax, normalize_sep=False, inset=False):

    config = [((0, 20), param_alpha, '-'), ((40, 60), param_alpha, ':'), ((80, 100), param_alpha, '-.')]

    for group, alpha, style in config:
        _figure_rank_move_of_group(network, ax, group, alpha=alpha, style=style, normalize_sep=normalize_sep)

    if inset:

        ax_inset = inset_axes(ax, width="45%", height="45%", loc='upper right')
        ax_inset.set_xlim(-1, 1)
        ax_inset.set_ylim(0, 0.75)

        _figure_rank_move_3group(network, ax_inset, normalize_sep=True)


def _figure_rank_move_of_group(network, ax, percent, alpha=param_alpha, style='-', normalize_sep=False):
     
    percent_low = percent[0]
    percent_high = percent[1]

    rank_moves = []

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
    
    bins = np.linspace(-1, 1, 40)

    for move in list(network.net.edges):

        r_move = rank_move(move[0], move[1], network)

        if (r_move is not None):

            rank_moves.append(r_move / rank_length)

    x = (bins[:-1] + bins[1:]) / 2
    ratio = len(rank_moves) / network.net.number_of_edges()

    hist, _ = np.histogram(rank_moves, bins=bins)
    normalized_hist = hist / np.sum(hist)

    (x_interp, y_interp) = _interhistogram(x, normalized_hist)

    if not normalize_sep:
        y_interp = [y * ratio for y in y_interp]

    ax.plot(x_interp, y_interp, color=network.color, linewidth=5, alpha=alpha, linestyle=style)


def figure_rank_move_phdnation():

    fig_path = "./fig/rank_move_phdnation.pdf"

    network_dict = facsimlib.processing.construct_network(net_type='global')

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0, 0.51, 0.25), fontsize=param_tick_size)

    _figure_rank_move_phdnation(network_dict["Biology"], ax[0], normalize_sep=True)
    _figure_rank_move_phdnation(network_dict["Computer Science"], ax[1], normalize_sep=True)
    _figure_rank_move_phdnation(network_dict["Physics"], ax[2], normalize_sep=True)

    x_label = "Relative Movement of Faculty (Normalized)"
    y_label = "Density"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=5),
               Line2D([0], [0], color='black', linestyle=':', linewidth=5),
               Patch(facecolor=network_dict["Biology"].color, label="Biology", alpha=param_alpha,
                     edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=network_dict["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ['Domestic', 'Global',
              network_dict["Biology"].name, network_dict["Computer Science"].name, network_dict["Physics"].name]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=6, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _figure_rank_move_phdnation(network, ax, normalize_sep=False):

    config = [('domestic', param_alpha, '-'), ('global', param_alpha, ':')]

    for group, alpha, style in config:
        _figure_rank_move_of_phdnation(network, ax, group, alpha=alpha, style=style, normalize_sep=normalize_sep)


def _figure_rank_move_of_phdnation(network, ax, phdnation, alpha=param_alpha, style='-', normalize_sep=False):
     
    if phdnation not in ['domestic', 'global']:
        return None

    rank_moves = []

    rank_length = network.rank_length

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")
    ns_global = NS('country_code', ["KR", "KOREA"], '!in', label="Korea")
    # ns_asia = NS('country_code', con_asia_without_kr, 'in', label="Asia")
    # ns_america = NS('country_code', con_america, 'in', label="America")
    # ns_europe = NS('country_code', con_europe, 'in', label="Europe")
    # ns_oceania = NS('country_code', con_ocenaia, 'in', label="Europe")

    if phdnation == 'domestic':
        ns = ns_korea

    else:
        ns = ns_global
    
    def rank_move(u_name, v_name, network: Field):

        u_name_norm = normalize_inst_name(u_name)
        v_name_norm = normalize_inst_name(v_name)

        ranks = network.ranks(inverse=True)

        u_rank = ranks[u_name_norm]
        v_rank = ranks[v_name_norm]

        if (all(rank is not None for rank in [u_rank, v_rank])):
            if (ns.hit(network.net.nodes[u_name])):
                return ranks[u_name_norm] - ranks[v_name_norm]
            
        return None
    
    bins = np.linspace(-1, 1, 40)

    for move in list(network.net.edges):

        r_move = rank_move(move[0], move[1], network)

        if (r_move is not None):

            rank_moves.append(r_move / rank_length)

    x = (bins[:-1] + bins[1:]) / 2
    ratio = len(rank_moves) / network.net.number_of_edges()

    hist, _ = np.histogram(rank_moves, bins=bins)
    normalized_hist = hist / np.sum(hist)

    (x_interp, y_interp) = _interhistogram(x, normalized_hist)

    if not normalize_sep:
        y_interp = [y * ratio for y in y_interp]

    ax.plot(x_interp, y_interp, color=network.color, linewidth=5, alpha=alpha, linestyle=style)


def figure_rank_move_region_with_inset(net_type='global'):

    fig_path = f"./fig/rank_move_region_{net_type}_inset.pdf"

    network_dict = facsimlib.processing.construct_network(net_type=net_type)

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(0, 0.25)
    plt.yticks(np.arange(0, 0.26, 0.125), fontsize=param_tick_size)

    _figure_rank_move_region(network_dict["Biology"], ax[0], inset=True)
    _figure_rank_move_region(network_dict["Computer Science"], ax[1], inset=True)
    _figure_rank_move_region(network_dict["Physics"], ax[2], inset=True)

    x_label = "Relative Movement of Faculty (Normalized)"
    y_label = "Density"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    for axi in ax.flatten():

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles1 = [Line2D([0], [0], color='black', linestyle='-', linewidth=5),
                Line2D([0], [0], color='black', linestyle=':', linewidth=5),
                Line2D([0], [0], color='black', linestyle='-.', linewidth=5),
                Line2D([0], [0], color='black', linestyle='--', linewidth=5),
                Line2D([0], [0], color='black', linestyle=(0, (4, 2, 2, 2)), linewidth=5)]
    
    handles2 = [Patch(facecolor=network_dict["Biology"].color, label="Biology",
                      alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                      alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=network_dict["Physics"].color, label="Physics",
                      alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels1 = ["Seoul", "Capital Area", "Metropolitan Cities", "Others", "-IST"]
    labels2 = [network_dict["Biology"].name, network_dict["Computer Science"].name, network_dict["Physics"].name]

    fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def figure_rank_move_region_by_pair(net_type='global'):

    fig_path = f"./fig/rank_move_region_{net_type}_by_pair.pdf"

    network_dict = facsimlib.processing.construct_network(net_type=net_type)

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(4, 3, sharey='all', figsize=(3 * param_fig_xsize, 4 * param_fig_ysize), dpi=200)

    plt.ylim(0, 0.125)
    plt.yticks(np.arange(0, 0.126, 0.125), fontsize=param_tick_size)
    
    regions = [NS('region', area_seoul, 'in', label="Seoul"),
               NS('region', area_capital, 'in', label="Capital Area"),
               NS('region', area_metro, 'in', label="Metropolitan\nCities"),
               NS('region', area_others, 'in', label="Others"),
               NS('name', inst_ists, 'in', label="-IST")]
        
    ist_style = (0, (2, 2, 4, 2))

    config = [(regions[0], param_alpha, '-'), (regions[1], param_alpha, ':'), (regions[2], param_alpha, '-.'),
              (regions[4], param_alpha, ist_style), (regions[3], param_alpha, '--')]
    
    for i in range(4):

        _figure_rank_move_region(network_dict["Biology"], ax[i, 0], inset=True, custom=[config[i], config[4]])
        _figure_rank_move_region(network_dict["Computer Science"], ax[i, 1], inset=True, custom=[config[i], config[4]])
        _figure_rank_move_region(network_dict["Physics"], ax[i, 2], inset=True, custom=[config[i], config[4]])

    x_label = "Relative Movement of Faculty (Normalized)"
    y_label = "Density"

    fig.supxlabel(x_label, fontsize=param_xlabel_size)
    fig.supylabel(y_label, fontsize=param_ylabel_size)

    def format_y(value, pos):
        return "{:.3f}".format(value)

    for axi in ax.flatten():

        axi.yaxis.set_major_formatter(ticker.FuncFormatter(format_y))

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=param_tick_size)

    handles1 = [Line2D([0], [0], color='black', linestyle='-', linewidth=5),
                Line2D([0], [0], color='black', linestyle=':', linewidth=5),
                Line2D([0], [0], color='black', linestyle='-.', linewidth=5),
                Line2D([0], [0], color='black', linestyle='--', linewidth=5),
                Line2D([0], [0], color='black', linestyle=ist_style, linewidth=5)]
    
    handles2 = [Patch(facecolor=network_dict["Biology"].color, label="Biology",
                      alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=network_dict["Computer Science"].color, label="Computer Science",
                      alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=network_dict["Physics"].color, label="Physics",
                      alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels1 = ["Seoul", "Capital Area", "Metropolitan Cities", "Others", "-IST"]
    labels2 = [network_dict["Biology"].name, network_dict["Computer Science"].name, network_dict["Physics"].name]

    fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.025), ncol=5, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _figure_rank_move_region(network, ax, normalize_sep=False, inset=False, custom=None):

    regions = [NS('region', area_seoul, 'in', label="Seoul"),
               NS('region', area_capital, 'in', label="Capital Area"),
               NS('region', area_metro, 'in', label="Metropolitan\nCities"),
               NS('region', area_others, 'in', label="Others"),
               NS('name', inst_ists, 'in', label="-IST")]
        
    ist_style = (0, (2, 2, 4, 2))

    if custom is None:

        config = [(regions[0], param_alpha, '-'), (regions[1], param_alpha, ':'), (regions[2], param_alpha, '-.'),
                  (regions[3], param_alpha, '--'), (regions[4], param_alpha, ist_style)]
        
    else:
        config = custom

    for ns, alpha, style in config:
        _figure_rank_move_of_ns(network, ax, ns, alpha=alpha, style=style, normalize_sep=normalize_sep)

    if inset:

        ax_inset = inset_axes(ax, width="45%", height="45%", loc='upper right')
        ax_inset.set_xlim(-1, 1)
        ax_inset.set_ylim(0, 0.75)

        _figure_rank_move_region(network, ax_inset, normalize_sep=True, custom=custom)


def _figure_rank_move_of_ns(network, ax, ns, alpha=param_alpha, style='-', normalize_sep=False):

    rank_moves = []

    rank_length = network.rank_length
    
    def rank_move(u_name, v_name, network: Field):

        u_name_norm = normalize_inst_name(u_name)
        v_name_norm = normalize_inst_name(v_name)

        ranks = network.ranks(inverse=True)

        u_rank = ranks[u_name_norm]
        v_rank = ranks[v_name_norm]

        if (all(rank is not None for rank in [u_rank, v_rank])):

            if ns.hit(network.net.nodes[u_name_norm]):
                return ranks[u_name_norm] - ranks[v_name_norm]
            
        return None
    
    bins = np.linspace(-1, 1, 40)

    for move in list(network.net.edges):

        r_move = rank_move(move[0], move[1], network)

        if (r_move is not None):

            rank_moves.append(r_move / rank_length)

    x = (bins[:-1] + bins[1:]) / 2
    ratio = len(rank_moves) / network.net.number_of_edges()

    hist, _ = np.histogram(rank_moves, bins=bins)
    normalized_hist = hist / np.sum(hist)

    (x_interp, y_interp) = _interhistogram(x, normalized_hist)

    if not normalize_sep:
        y_interp = [y * ratio for y in y_interp]

    ax.plot(x_interp, y_interp, color=network.color, linewidth=5, alpha=alpha, linestyle=style)
