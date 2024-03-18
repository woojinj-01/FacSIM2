import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import facsimlib.processing
import facsimlib.math
from facsimlib.plot.params import *
from facsimlib.plot.palette import split_color_by, palette_bio, palette_cs, palette_phy

palette_dict = {"Biology": palette_bio, "Domestic Biology": palette_bio,
                "Computer Science": palette_cs, "Domestic Computer Science": palette_cs,
                "Physics": palette_phy, "Domestic Physics": palette_phy}


def figure_hires(palette):

    assert palette in ['hatches', 'explicit', 'split']
    
    fig_path = f"./fig/hires_{palette}.pdf"

    networks_global = facsimlib.processing.construct_network()
    networks_domestic = facsimlib.processing.construct_network(net_type='domestic')

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(2, 2, figsize=(2 * param_fig_xsize, 2 * param_fig_ysize), dpi=200)

    _figure_hires_distro(networks_global, ax[0, 0], palette)
    _figure_hires_z(networks_global, ax[0, 1], palette)

    _figure_hires_distro(networks_domestic, ax[1, 0], palette)
    _figure_hires_z(networks_domestic, ax[1, 1], palette)

    if palette == 'hatches':

        handles1 = [Patch(facecolor=networks_global["Biology"].color, alpha=param_alpha, edgecolor='black', linewidth=3),
                    Patch(facecolor=networks_global["Computer Science"].color, alpha=param_alpha, edgecolor='black', linewidth=3),
                    Patch(facecolor=networks_global["Physics"].color, alpha=param_alpha, edgecolor='black', linewidth=3)]

        handles2 = [Patch(hatch='-', facecolor='white', alpha=param_alpha, edgecolor='black', linewidth=3),
                    Patch(hatch='o', facecolor='white', alpha=param_alpha, edgecolor='black', linewidth=3),
                    Patch(hatch='x', facecolor='white', alpha=param_alpha, edgecolor='black', linewidth=3)]

        labels1 = ["Biology", "Computer Science", "Physics"]
        labels2 = ["Up hires", "Self hires", "Down hires"]

        fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
        fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    else:

        if palette == 'explicit':

            pal_bio = palette_bio[0:3]
            pal_cs = palette_cs[0:3]
            pal_phy = palette_phy[0:3]
        
        else:
            pal_bio = split_color_by(networks_global["Biology"].color, 3)
            pal_cs = split_color_by(networks_global["Computer Science"].color, 3)
            pal_phy = split_color_by(networks_global["Physics"].color, 3)

        handles_bio = [Patch(facecolor=pal_bio[i], alpha=param_alpha, edgecolor='black', linewidth=3) for i in range(len(pal_bio))]
        handles_cs = [Patch(facecolor=pal_cs[i], alpha=param_alpha, edgecolor='black', linewidth=3) for i in range(len(pal_cs))]
        handles_phy = [Patch(facecolor=pal_phy[i], alpha=param_alpha, edgecolor='black', linewidth=3) for i in range(len(pal_phy))]
        
        labels_root = ["Up-hires", "Self-hires", "Down-hires"]

        labels_bio = [f"Biology ({root})" for root in labels_root]
        labels_cs = [f"Computer Science ({root})" for root in labels_root]
        labels_phy = [f"Physics ({root})" for root in labels_root]

        handles = [h_field[i] for i in range(3) for h_field in [handles_bio, handles_cs, handles_phy]]
        labels = [l_field[i] for i in range(3) for l_field in [labels_bio, labels_cs, labels_phy]]

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def figure_hires_using_colors():

    print("This function is deprecated")
    assert False
    
    fig_path = "./fig/hires_colors.pdf"

    networks_global = facsimlib.processing.construct_network()
    networks_domestic = facsimlib.processing.construct_network(net_type='domestic')

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(2, 2, figsize=(2 * param_fig_xsize, 2 * param_fig_ysize), dpi=200)

    _figure_hires_distro(networks_global, ax[0, 0], using_hatches=False)
    _figure_hires_z(networks_global, ax[0, 1], using_hatches=False)

    _figure_hires_distro(networks_domestic, ax[1, 0], using_hatches=False)
    _figure_hires_z(networks_domestic, ax[1, 1], using_hatches=False)

    # handles1 = [Patch(facecolor=networks_global["Biology"].color, alpha=param_alpha, edgecolor='black', linewidth=3),
    #             Patch(facecolor=networks_global["Computer Science"].color, alpha=param_alpha, edgecolor='black', linewidth=3),
    #             Patch(facecolor=networks_global["Physics"].color, alpha=param_alpha, edgecolor='black', linewidth=3)]
    
    # handles2 = [Patch(hatch='-', facecolor='white', alpha=param_alpha, edgecolor='black', linewidth=3),
    #             Patch(hatch='o', facecolor='white', alpha=param_alpha, edgecolor='black', linewidth=3),
    #             Patch(hatch='x', facecolor='white', alpha=param_alpha, edgecolor='black', linewidth=3)]

    # labels1 = ["Biology", "Computer Science", "Physics"]
    # labels2 = ["Up hires", "Self hires", "Down hires"]

    # fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    # fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _figure_hires_distro(network_dict, ax, palette):

    assert palette in ['hatches', 'explicit', 'split']

    uhires = {}
    self_hires = {}
    down_hires = {}

    for net in network_dict.values():

        net.set_ranks()

        stat = facsimlib.math.up_down_hires(net, normalized=True)

        uhires[net.name] = stat[0]
        self_hires[net.name] = stat[1]
        down_hires[net.name] = stat[2]

    y_label = "Hires (Normalized)"

    ax.set_ylabel(y_label)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    for net in network_dict.values():

        if len(net.name.split(' ')) >= 2:
            name_to_put = net.name.replace(' ', '\n')
        else:
            name_to_put = net.name

        if palette == 'hatches':

            ax.bar(name_to_put, down_hires[net.name], color=net.color, hatch='x', alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(name_to_put, self_hires[net.name], color=net.color, bottom=down_hires[net.name], hatch='o',
                   alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(name_to_put, uhires[net.name], color=net.color, bottom=down_hires[net.name] + self_hires[net.name],
                   hatch='-', alpha=param_alpha, edgecolor='black', linewidth=2)
            
            continue
        
        elif palette == 'explicit':
            palette = palette_dict[net.name][0:3]

        else:
            palette = split_color_by(net.color, 3)

        ax.bar(name_to_put, down_hires[net.name], color=palette[0], alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.bar(name_to_put, self_hires[net.name], color=palette[1], bottom=down_hires[net.name],
               alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.bar(name_to_put, uhires[net.name], color=palette[2], bottom=down_hires[net.name] + self_hires[net.name],
               alpha=param_alpha, edgecolor='black', linewidth=2)
            

def _figure_hires_z(network_dict, ax, palette):

    assert palette in ['hatches', 'explicit', 'split']

    trial = 500

    uhires = {}
    self_hires = {}
    down_hires = {}

    for net in network_dict.values():

        net.set_ranks()

        stat = facsimlib.math.up_down_hires(net, normalized=True)

        network_rand = net.randomize(trial)

        urand = []
        se_rand = []
        do_rand = []

        for net_rand in network_rand:

            net_rand.set_ranks()

            (up, se, do) = facsimlib.math.up_down_hires(net_rand, normalized=True)

            urand.append(up)
            se_rand.append(se)
            do_rand.append(do)

        uz = (stat[0] - np.mean(urand)) / np.std(urand)
        se_z = (stat[1] - np.mean(se_rand)) / np.std(se_rand)
        do_z = (stat[2] - np.mean(do_rand)) / np.std(do_rand)

        uhires[net.name] = uz
        self_hires[net.name] = se_z
        down_hires[net.name] = do_z

    y_label = "Z-Score"
    ax.set_ylabel(y_label)

    ax.set_ylim(-10, 15)
    ax.set_yticks(range(-10, 16, 5))

    bar_width = 0.25
    x_pos = 0

    for net in network_dict.values():

        x_pos_1 = x_pos
        x_pos_2 = x_pos + bar_width
        x_pos_3 = x_pos + 2 * bar_width

        if palette == 'hatches':

            ax.bar(x_pos_1, uhires[net.name], width=bar_width, color=net.color,
                   hatch='-', alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(x_pos_2, self_hires[net.name], width=bar_width, color=net.color,
                   hatch='o', alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(x_pos_3, down_hires[net.name], width=bar_width, color=net.color,
                   hatch='x', alpha=param_alpha, edgecolor='black', linewidth=2)
            
        else:
            if palette == 'explicit':
                palette = palette_dict[net.name][0:3]

            else:
                palette = split_color_by(net.color, 3)

            ax.bar(x_pos_1, uhires[net.name], width=bar_width, color=palette[0],
                   alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(x_pos_2, self_hires[net.name], width=bar_width, color=palette[1],
                   alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(x_pos_3, down_hires[net.name], width=bar_width, color=palette[2],
                   alpha=param_alpha, edgecolor='black', linewidth=2)

        x_pos += 1

    x_ticks = []

    for net_name in ["Biology", "Computer Science", "Physics"]:

        net = network_dict[net_name]

        if len(net.name.split(' ')) >= 2:
            x_ticks.append(net.name.replace(' ', '\n'))
        else:
            x_ticks.append(net.name)

    ax.set_xticks([x + bar_width for x in range(3)], x_ticks)
    ax.axhline(0, color='black', linewidth=1)


if __name__ == "__main__":

    figure_hires('hatches')
    figure_hires('explicit')
    figure_hires('split')