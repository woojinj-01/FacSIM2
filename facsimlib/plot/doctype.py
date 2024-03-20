import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import math

import facsimlib.processing
import facsimlib.math
from facsimlib.academia import NodeSelect as NS
from facsimlib.text import get_country_code, area_seoul, area_capital, area_metro, area_others, \
    con_america, con_europe, con_ocenaia, con_asia_without_kr, inst_ists
from facsimlib.plot.params import *
from facsimlib.plot.palette import split_color_by, palette_bio, palette_cs, palette_phy
from facsimlib.plot.general import process_gini_coeff, sample_from_data

palette_dict = {"Biology": palette_bio, "Computer Science": palette_cs, "Physics": palette_phy}

explicit_alpha = 1


def doctorate_group(palette='explicit'):

    assert palette in ['hatches', 'explicit', 'split']

    fig_path = f"./fig/doctorate_group_{palette}.pdf"

    if palette == 'explicit':
        alpha_to_use = explicit_alpha
    else:
        alpha_to_use = param_alpha

    networks_dict = facsimlib.processing.construct_network()
    
    for net in networks_dict.values():
        net.copy_ranks_from(net.domestic.set_ranks())

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200, sharey=True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    
    plt.yticks(range(1, 11), fontsize=param_tick_size)
    plt.gca().invert_yaxis()

    x_label = "Number of Assistant Professors (Normalized)"
    y_label = "Group"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    pal_bio = _doctorate_group(networks_dict["Biology"], ax[0], palette)
    pal_cs = _doctorate_group(networks_dict["Computer Science"], ax[1], palette)
    pal_phy = _doctorate_group(networks_dict["Physics"], ax[2], palette)

    if palette == 'hatches':

        handles1 = [Patch(facecolor=networks_dict["Biology"].color, label="Biology", alpha=alpha_to_use, edgecolor='black', linewidth=3),
                    Patch(facecolor=networks_dict["Computer Science"].color, label="Computer Science", alpha=alpha_to_use, edgecolor='black', linewidth=3),
                    Patch(facecolor=networks_dict["Physics"].color, label="Physics", alpha=alpha_to_use, edgecolor='black', linewidth=3)]
        
        handles2 = [Patch(edgecolor='black', hatch='*', facecolor='white'),
                    Patch(edgecolor='black', hatch='O', facecolor='white'),
                    Patch(edgecolor='black', hatch='+', facecolor='white'),
                    Patch(edgecolor='black', hatch='/', facecolor='white')]

        labels1 = ["Biology", "Computer Science", "Physics"]
        labels2 = ["America", "Asia & Oceania", "Europe", "South Korea"]

        fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
        fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.045), ncol=5, frameon=False)

    else:

        handles_bio = [Patch(facecolor=pal_bio[i], alpha=alpha_to_use, edgecolor='black', linewidth=3) for i in range(len(pal_bio))]
        handles_cs = [Patch(facecolor=pal_cs[i], alpha=alpha_to_use, edgecolor='black', linewidth=3) for i in range(len(pal_cs))]
        handles_phy = [Patch(facecolor=pal_phy[i], alpha=alpha_to_use, edgecolor='black', linewidth=3) for i in range(len(pal_phy))]
        
        labels_root = ["America", "Asia & Oceania", "Europe", "South Korea"]

        labels_bio = [f"Biology ({root})" for root in labels_root]
        labels_cs = [f"Computer Science ({root})" for root in labels_root]
        labels_phy = [f"Physics ({root})" for root in labels_root]

        handles = [h_field[i] for i in range(4) for h_field in [handles_bio, handles_cs, handles_phy]]
        labels = [l_field[i] for i in range(4) for l_field in [labels_bio, labels_cs, labels_phy]]

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=4, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _doctorate_group(network, ax, palette):
    
    assert palette in ['hatches', 'explicit', 'split']

    if palette == 'explicit':
        alpha_to_use = explicit_alpha
    else:
        alpha_to_use = param_alpha

    num_group = 10

    total_count = {}

    kr_count = {}
    asia_oceania__count = {}
    america_count = {}
    europe_count = {}

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")
    ns_asia = NS('country_code', con_asia_without_kr, 'in', label="Asia")
    ns_america = NS('country_code', con_america, 'in', label="America")
    ns_europe = NS('country_code', con_europe, 'in', label="Europe")
    ns_oceania = NS('country_code', con_ocenaia, 'in', label="Europe")

    for node in network.net.nodes:

        edges_in = network.net.in_edges(node, data=True)

        rank = network.inst(node)['rank']

        if (rank is None):
            continue

        total_count[rank] = len(edges_in)
        kr_count[rank] = 0
        asia_oceania__count[rank] = 0
        america_count[rank] = 0
        europe_count[rank] = 0

        for src, dst, data in edges_in:

            if (ns_korea.hit(network.net.nodes[src])):
                kr_count[rank] += 1

            elif ((ns_asia.hit(network.net.nodes[src]))):
                asia_oceania__count[rank] += 1

            elif ((ns_america.hit(network.net.nodes[src]))):
                america_count[rank] += 1

            elif ((ns_europe.hit(network.net.nodes[src]))):
                europe_count[rank] += 1

            elif ((ns_oceania.hit(network.net.nodes[src]))):
                asia_oceania__count[rank] += 1

    ranks_sorted = [key for key, _ in sorted(total_count.items())]
    kr_count_sorted = [value for _, value in sorted(kr_count.items())]
    asia_oceania_count_sorted = [value for _, value in sorted(asia_oceania__count.items())]
    america_count_sorted = [value for _, value in sorted(america_count.items())]
    europe_count_sorted = [value for _, value in sorted(europe_count.items())]

    groupsize = np.floor(len(ranks_sorted) / num_group)

    if (groupsize == 1):

        x_co = ranks_sorted
        y_co_kr = kr_count_sorted
        y_co_asia_oceania = asia_oceania_count_sorted
        y_co_america = america_count_sorted
        y_co_europe = europe_count_sorted

    else:
        
        x_co = []

        y_co_kr = []
        y_co_asia_oceania = []
        y_co_america = []
        y_co_europe = []
        
        index = 0
        groupid = 1

        while index < num_group * groupsize:

            elements_kr = []
            elements_asia_oceania = []
            elements_america = []
            elements_europe = []

            while (groupsize > len(elements_kr)):

                elements_kr.append(kr_count_sorted[index])
                elements_asia_oceania.append(asia_oceania_count_sorted[index])
                elements_america.append(america_count_sorted[index])
                elements_europe.append(europe_count_sorted[index])

                index += 1

                if (index >= len(ranks_sorted)):
                    break

            x_co.append(groupid)

            y_co_kr.append(sum(elements_kr))
            y_co_asia_oceania.append(sum(elements_asia_oceania))
            y_co_america.append(sum(elements_america))
            y_co_europe.append(sum(elements_europe))

            groupid += 1

        while index < len(ranks_sorted):

            y_co_kr[-1] += kr_count_sorted[index]
            y_co_asia_oceania[-1] += asia_oceania_count_sorted[index]
            y_co_america[-1] += america_count_sorted[index]
            y_co_europe[-1] += europe_count_sorted[index]

            index += 1

    def sum_of_the_lists(i):
        return y_co_kr[i] + y_co_asia_oceania[i] + y_co_america[i] + y_co_europe[i]

    y_co_kr_norm = [y_co_kr[i] / sum_of_the_lists(i) if sum_of_the_lists(i) != 0 else 0 for i in range(len(x_co))]
    y_co_asia_oceania_norm = [y_co_asia_oceania[i] / sum_of_the_lists(i) if sum_of_the_lists(i) != 0 else 0 for i in range(len(x_co))]
    y_co_america_norm = [y_co_america[i] / sum_of_the_lists(i) if sum_of_the_lists(i) != 0 else 0 for i in range(len(x_co))]
    y_co_europe_norm = [y_co_europe[i] / sum_of_the_lists(i) if sum_of_the_lists(i) != 0 else 0 for i in range(len(x_co))]

    y_co_kr = y_co_kr_norm
    y_co_asia_oceania = y_co_asia_oceania_norm
    y_co_america = y_co_america_norm
    y_co_europe = y_co_europe_norm

    ax.tick_params(axis='both', which='major', labelsize=param_tick_size)

    ax.set_xlim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))

    if palette == 'hatches':

        ax.barh(x_co, y_co_america, color=network.color,
                hatch='*', alpha=alpha_to_use, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_asia_oceania, color=network.color, left=y_co_america,
                hatch='O', alpha=alpha_to_use, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_europe, color=network.color, left=[y_co_asia_oceania[i] + y_co_america[i] for i in range(len(y_co_asia_oceania))],
                hatch='+', alpha=alpha_to_use, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_kr, color=network.color, left=[y_co_asia_oceania[i] + y_co_america[i] + y_co_europe[i] for i in range(len(y_co_asia_oceania))],
                hatch='/', alpha=alpha_to_use, edgecolor='black', linewidth=2)
        
        return None
    
    elif palette == 'explicit':

        explicit_palette_chosen = palette_dict[network.name]
        palette = explicit_palette_chosen[:2] + explicit_palette_chosen[3:]
        
    elif palette == 'split':
        palette = split_color_by(network.color, 4)
    
    else:
        return None

    ax.barh(x_co, y_co_america, color=palette[0], alpha=alpha_to_use, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_asia_oceania, color=palette[1], left=y_co_america, alpha=alpha_to_use, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_europe, color=palette[2], left=[y_co_asia_oceania[i] + y_co_america[i] for i in range(len(y_co_asia_oceania))],
            alpha=alpha_to_use, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_kr, color=palette[3], left=[y_co_asia_oceania[i] + y_co_america[i] + y_co_europe[i] for i in range(len(y_co_asia_oceania))],
            alpha=alpha_to_use, edgecolor='black', linewidth=2)

    return palette
    

def lorentz_curve_group():
    
    fig_path = "./fig/lorentz_curve_group.pdf"

    networks_dict = facsimlib.processing.construct_network()

    for net in networks_dict.values():
        net.copy_ranks_from(net.domestic.set_ranks())

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200, sharey=True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 25), fontsize=param_tick_size)

    x_label = "Cumulative Ratio of Groups (%)"
    y_label = "Cumulative Ratio of In Degrees (%)"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    _lorentz_curve_group(networks_dict["Biology"], ax[0])
    _lorentz_curve_group(networks_dict["Computer Science"], ax[1])
    _lorentz_curve_group(networks_dict["Physics"], ax[2])

    handles1 = [Patch(facecolor=networks_dict["Biology"].color, label="Biology", alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=networks_dict["Computer Science"].color, label="Computer Science", alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=networks_dict["Physics"].color, label="Physics", alpha=param_alpha, edgecolor='black', linewidth=3)]
    
    handles2 = [Line2D([0], [0], color='black', linestyle=':', linewidth=5),
                Line2D([0], [0], color='black', linestyle='-', linewidth=5),]
    
    handles1.extend(handles2)

    labels = ["Biology", "Computer Science", "Physics", "America", "South Korea"]

    fig.legend(handles1, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _lorentz_curve_group(network, ax):

    div_10per = [i for i in range(0, 101, 20)]

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))

    data_america = []
    data_korea = []
    ranks = network.ranks(inverse=True)

    target = [data_america, data_korea]
    info = [("America", "dashed"), ("Korea", "solid")]

    for node in network.net.nodes:

        if ranks[node] is None:
            continue

        edges_in = network.net.in_edges(node, data=True)

        num_america = 0
        num_korea = 0

        for src, dst, data in edges_in:
            
            if get_country_code(data['u_name']) in con_america:
                num_america += 1

            elif (get_country_code(data['u_name']) in ["KR", "KOREA"]):
                num_korea += 1

        data_america.append(num_america)
        data_korea.append(num_korea)

    for data_con, info_con in zip(target, info):

        (label, style) = info_con

        (gini_coeff, x_co, y_co, base_co) = process_gini_coeff(data_con)

        ax.plot(x_co, y_co, color=network.color, linewidth=3, alpha=param_alpha, linestyle=style)
        ax.scatter(div_10per, [sample_from_data(x_co, y_co, index) for index in div_10per],
                   c=network.color, s=30, clip_on=False, alpha=1, marker='o')

        ax.plot(base_co, base_co, color='black', linewidth=0.5)

        if label == "Korea":
            ax.fill_between(x_co, y_co, base_co, color='grey', alpha=0.1)

            sum_1 = 0
            sum_2 = 0

            for x, y in zip(x_co, y_co):
                
                sum_1 += x * (x - y)
                sum_2 += x - y

            centroid_x = (sum_1 / sum_2)

            for i in range(len(x_co)):

                if x_co[i] < centroid_x < x_co[i + 1]:
                    centroid_x_index = i

            centroid_y = (x_co[centroid_x_index] + y_co[centroid_x_index]) / 2

            text_to_put = "%.3f" % (math.floor(gini_coeff * 1000) / 1000)

            ax.text(centroid_x / 100, centroid_y / 100, text_to_put, transform=ax.transAxes,
                    ha='center', va='center', fontsize=param_pannel_size, fontweight='bold')
            

def doctorate_region(palette='explicit'):

    assert palette in ['hatches', 'explicit', 'split']
    
    fig_path = f"./fig/doctorate_region_{palette}.pdf"

    if palette == 'explicit':
        alpha_to_use = explicit_alpha
    else:
        alpha_to_use = param_alpha

    networks_dict = facsimlib.processing.construct_network()
    
    for net in networks_dict.values():
        net.copy_ranks_from(net.domestic.set_ranks())

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200, sharey=True)
    plt.gca().invert_yaxis()

    x_label = "Number of Assistant Professors (Normalized)"

    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    pal_bio = _doctorate_region(networks_dict["Biology"], ax[0], palette=palette)
    pal_cs = _doctorate_region(networks_dict["Computer Science"], ax[1], palette=palette)
    pal_phy = _doctorate_region(networks_dict["Physics"], ax[2], palette=palette)

    if palette == 'hatches':

        handles1 = [Patch(facecolor=networks_dict["Biology"].color, label="Biology", alpha=alpha_to_use, edgecolor='black', linewidth=3),
                    Patch(facecolor=networks_dict["Computer Science"].color, label="Computer Science", alpha=alpha_to_use, edgecolor='black', linewidth=3),
                    Patch(facecolor=networks_dict["Physics"].color, label="Physics", alpha=alpha_to_use, edgecolor='black', linewidth=3)]
        
        handles2 = [Patch(edgecolor='black', hatch='*', facecolor='white'),
                    Patch(edgecolor='black', hatch='O', facecolor='white'),
                    Patch(edgecolor='black', hatch='+', facecolor='white'),
                    Patch(edgecolor='black', hatch='-', facecolor='white'),
                    Patch(edgecolor='black', hatch='/', facecolor='white')]

        labels1 = ["Biology", "Computer Science", "Physics"]
        labels2 = ["America", "Asia", "Europe", "Oceania", "South Korea"]

        fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
        fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.045), ncol=5, frameon=False)

    else:

        handles_bio = [Patch(facecolor=pal_bio[i], alpha=alpha_to_use, edgecolor='black', linewidth=3) for i in range(len(pal_bio))]
        handles_cs = [Patch(facecolor=pal_cs[i], alpha=alpha_to_use, edgecolor='black', linewidth=3) for i in range(len(pal_cs))]
        handles_phy = [Patch(facecolor=pal_phy[i], alpha=alpha_to_use, edgecolor='black', linewidth=3) for i in range(len(pal_phy))]
        
        labels_root = ["America", "Asia & Oceania", "Europe", "South Korea"]

        labels_bio = [f"Biology ({root})" for root in labels_root]
        labels_cs = [f"Computer Science ({root})" for root in labels_root]
        labels_phy = [f"Physics ({root})" for root in labels_root]

        handles = [h_field[i] for i in range(4) for h_field in [handles_bio, handles_cs, handles_phy]]
        labels = [l_field[i] for i in range(4) for l_field in [labels_bio, labels_cs, labels_phy]]

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=4, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _doctorate_region(network, ax, palette):

    assert palette in ['hatches', 'explicit', 'split']

    if palette == 'explicit':
        alpha_to_use = explicit_alpha
    else:
        alpha_to_use = param_alpha

    ns_list = [NS('region', area_seoul, 'in', label="Seoul"),
               NS('region', area_capital, 'in', label="Capital Area"),
               NS('region', area_metro, 'in', label="Metropolitan\nCities"),
               NS('region', area_others, 'in', label="Others"),
               NS('name', inst_ists, 'in', label="-IST")]
    
    total_count = {}

    kr_count = {}
    asia_oceania_count = {}
    america_count = {}
    europe_count = {}

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")
    ns_asia = NS('country_code', con_asia_without_kr, 'in', label="Asia")
    ns_america = NS('country_code', con_america, 'in', label="America")
    ns_europe = NS('country_code', con_europe, 'in', label="Europe")
    ns_oceania = NS('country_code', con_ocenaia, 'in', label="Europe")

    for ns in ns_list:

        total_count[ns.label] = 0
        kr_count[ns.label] = 0
        asia_oceania_count[ns.label] = 0
        america_count[ns.label] = 0
        europe_count[ns.label] = 0

    for node in network.net.nodes():

        ns_list_hit = []

        for ns in ns_list[::-1]:
            if ns.hit(network.net.nodes[node]):
                ns_list_hit.append(ns.label)
                break

        edges_in = network.net.in_edges(node, data=True)

        for src, dst, data in edges_in:

            for ns_repr in ns_list_hit:
                total_count[ns_repr] += 1

            if (get_country_code(data['u_name']) == 'KR'):

                for ns_repr in ns_list_hit:
                    kr_count[ns_repr] += 1

            if (ns_korea.hit(network.net.nodes[src])):
                for ns_repr in ns_list_hit:
                    kr_count[ns_repr] += 1

            elif ((ns_asia.hit(network.net.nodes[src]))):
                for ns_repr in ns_list_hit:
                    asia_oceania_count[ns_repr] += 1

            elif ((ns_america.hit(network.net.nodes[src]))):
                for ns_repr in ns_list_hit:
                    america_count[ns_repr] += 1

            elif ((ns_europe.hit(network.net.nodes[src]))):
                for ns_repr in ns_list_hit:
                    europe_count[ns_repr] += 1

            elif ((ns_oceania.hit(network.net.nodes[src]))):
                for ns_repr in ns_list_hit:
                    asia_oceania_count[ns_repr] += 1

    total_count_sorted = [value for _, value in sorted(total_count.items())]

    x_co = [ns.label if ns.label is not None else ns.__repr__() for ns in ns_list]

    y_co_kr = [value for _, value in sorted(kr_count.items())]
    y_co_asia_oceania = [value for _, value in sorted(asia_oceania_count.items())]
    y_co_america = [value for _, value in sorted(america_count.items())]
    y_co_europe = [value for _, value in sorted(europe_count.items())]
            
    y_co_kr_norm = [y_co_kr[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]
    y_co_asia_oceania_norm = [y_co_asia_oceania[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]
    y_co_america_norm = [y_co_america[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]
    y_co_europe_norm = [y_co_europe[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]

    y_co_kr = y_co_kr_norm
    y_co_asia_oceania = y_co_asia_oceania_norm
    y_co_america = y_co_america_norm
    y_co_europe = y_co_europe_norm

    ax.set_xlim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))

    if palette == 'hatches':
    
        ax.barh(x_co, y_co_america, color=network.color,
                hatch='*', alpha=alpha_to_use, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_asia_oceania, color=network.color, left=y_co_america, 
                hatch='O', alpha=alpha_to_use, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_europe, color=network.color, left=[y_co_asia_oceania[i] + y_co_america[i] for i in range(len(y_co_asia_oceania))],
                hatch='+', alpha=alpha_to_use, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_kr, color=network.color, left=[y_co_asia_oceania[i] + y_co_america[i] + y_co_europe[i] for i in range(len(y_co_asia_oceania))],
                hatch='/', alpha=alpha_to_use, edgecolor='black', linewidth=2)
        
        return None

    elif palette == 'explicit':
        explicit_palette_chosen = palette_dict[network.name]
        palette = explicit_palette_chosen[:2] + explicit_palette_chosen[3:]
    
    elif palette == 'split':
        palette = split_color_by(network.color, 4)

    else:
        return None

    ax.barh(x_co, y_co_america, color=palette[0], alpha=alpha_to_use, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_asia_oceania, color=palette[1], left=y_co_america,
            alpha=alpha_to_use, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_europe, color=palette[2], left=[y_co_asia_oceania[i] + y_co_america[i] for i in range(len(y_co_asia_oceania))],
            alpha=alpha_to_use, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_kr, color=palette[3], left=[y_co_asia_oceania[i] + y_co_america[i] + y_co_europe[i] for i in range(len(y_co_asia_oceania))],
            alpha=alpha_to_use, edgecolor='black', linewidth=2)
    
    return palette


def lorentz_curve_region():
    
    fig_path = "./fig/lorentz_curve_region.pdf"

    networks_dict = facsimlib.processing.construct_network()

    for net in networks_dict.values():
        net.copy_ranks_from(net.domestic.set_ranks())

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200, sharey=True)
    
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 25), fontsize=param_tick_size)

    x_label = "Cumulative Ratio of Groups (%)"
    y_label = "Cumulative Ratio of In Degrees (%)"

    ax[0].set_ylabel(y_label, fontsize=param_ylabel_size)
    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    _lorentz_curve_region(networks_dict["Biology"], ax[0])
    _lorentz_curve_region(networks_dict["Computer Science"], ax[1])
    _lorentz_curve_region(networks_dict["Physics"], ax[2])

    handles1 = [Patch(facecolor=networks_dict["Biology"].color, label="Biology", alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=networks_dict["Computer Science"].color, label="Computer Science", alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=networks_dict["Physics"].color, label="Physics", alpha=param_alpha, edgecolor='black', linewidth=3)]
    
    handles2 = [Line2D([0], [0], color='black', linestyle=':', linewidth=5),
                Line2D([0], [0], color='black', linestyle='-', linewidth=5),]
    
    handles1.extend(handles2)

    labels = ["Biology", "Computer Science", "Physics", "America", "South Korea"]

    fig.legend(handles1, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _lorentz_curve_region(network, ax):
    
    ns_seoul = NS('region', area_seoul, 'in', label="Seoul")
    ns_cap = NS('region', area_capital, 'in', label="Capital Area")
    ns_metro = NS('region', area_metro, 'in', label="Metropolitan\nCities")
    ns_others = NS('region', area_others, 'in', label="Others")
    ns_ist = NS('name', inst_ists, 'in', label="-IST")

    div_10per = [i for i in range(0, 101, 20)]

    data_america = [0, 0, 0, 0, 0, 0]
    data_korea = [0, 0, 0, 0, 0, 0]

    ind_seoul = 0
    ind_ist = 1
    ind_cap = 2
    ind_metro = 3
    ind_others = 4

    target = [data_america, data_korea]
    info = [("America", "dashed"), ("Korea", "solid")]

    for node in network.net.nodes:

        edges_in = network.net.in_edges(node, data=True)

        if ns_ist.hit(network.net.nodes[node]):
            ind = ind_ist

        elif ns_seoul.hit(network.net.nodes[node]):
            ind = ind_seoul

        elif ns_cap.hit(network.net.nodes[node]):
            ind = ind_cap

        elif ns_metro.hit(network.net.nodes[node]):
            ind = ind_metro

        elif ns_others.hit(network.net.nodes[node]):
            ind = ind_others

        else:
            continue

        for src, dst, data in edges_in:
            
            if get_country_code(data['u_name']) in con_america:
                data_america[ind] += 1

            elif (get_country_code(data['u_name']) in ["KR", "KOREA"]):
                data_korea[ind] += 1

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))

    for data_con, info_con in zip(target, info):

        (label, style) = info_con

        (gini_coeff, x_co, y_co, base_co) = process_gini_coeff(data_con)

        ax.plot(x_co, y_co, color=network.color, linewidth=2, label=label, alpha=param_alpha, linestyle=style)
        ax.scatter(div_10per, [sample_from_data(x_co, y_co, index) for index in div_10per],
                   c=network.color, s=30, clip_on=False, alpha=1, marker='o')

        ax.plot(base_co, base_co, color='black', linewidth=0.5)

        def _put_gini_coeff(offset):

            ax.fill_between(x_co, y_co, base_co, color='grey', alpha=0.1)

            sum_1 = 0
            sum_2 = 0

            for x, y in zip(x_co, y_co):
                
                sum_1 += x * (x - y)
                sum_2 += x - y

            centroid_x = (sum_1 / sum_2)

            for i in range(len(x_co)):

                if x_co[i] < centroid_x < x_co[i + 1]:
                    centroid_x_index = i

            centroid_y = (x_co[centroid_x_index] + y_co[centroid_x_index]) / 2

            text_to_put = "%.3f" % (math.floor(gini_coeff * 1000) / 1000)

            ax.text(centroid_x / 100, centroid_y / 100 + offset, text_to_put, transform=ax.transAxes,
                    ha='center', va='center', fontsize=param_pannel_size, fontweight='bold')

        if network.name == "Computer Science":
            if label == "America":
                _put_gini_coeff(0.13)
        else:
            if label == "Korea":
                _put_gini_coeff(0.1)


if __name__ == "__main__":
    
    # doctorate_group(palette='hatches')
    doctorate_group(palette='explicit')
    # doctorate_group(palette='split')

    # doctorate_region(palette='hatches')
    doctorate_region(palette='explicit')
    # doctorate_region(palette='split')

