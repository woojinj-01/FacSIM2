import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import numpy as np
import math
from pdf2image import convert_from_path
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.interpolate import interp1d
import colorsys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import facsimlib.processing
import facsimlib.math
from facsimlib.academia import Field, NodeSelect as NS
from facsimlib.text import get_country_code, normalize_inst_name, area_seoul, area_capital, area_metro, area_others, \
    con_america, con_europe, con_ocenaia, con_asia_without_kr, inst_ists

param_scale_factor = 2

param_fig_xsize = 10
param_fig_ysize = 10
param_alpha = 0.5

param_font_size = 26
param_xlabel_size = 30 * 1.5
param_ylabel_size = 30 * 1.5
param_tick_size = 30 * 1.5
param_legend_size = 50
param_pannel_size = param_legend_size

param_font = {'family': 'Helvetica Neue', 'size': param_font_size}


def hex_to_rgb(hex_color):

    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:], 16)

    return (red, green, blue)


def rgb_to_hex(rgb):

    red = min(max(rgb[0], 0), 255)
    green = min(max(rgb[1], 0), 255)
    blue = min(max(rgb[2], 0), 255)

    hex_color = "#{:02X}{:02X}{:02X}".format(red, green, blue)
    
    return hex_color


def hex_to_hsv(hex_color):

    rgb_color = hex_to_rgb(hex_color)

    hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
    
    return hsv_color


def hsv_to_hex(hsv_color):

    rgb_color = colorsys.hsv_to_rgb(hsv_color[0], hsv_color[1], hsv_color[2])
    
    hex_color = "#{:02X}{:02X}{:02X}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
    
    return hex_color


def split_color_by(hex_color, division=5):

    hsv_origin = hex_to_hsv(hex_color)

    base_saturation = 0.3

    delta = (hsv_origin[1] - base_saturation) / (division - 1)

    hex_colors = [hsv_to_hex((hsv_origin[0], hsv_origin[1] - delta * i, hsv_origin[2])) for i in range(0, division)]

    return hex_colors


def process_gini_coeff(data):

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


def sample_from_data(x_co, y_co, x_co_sample):

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


def figure_1():

    figure_lorentz_curve()


def figure_2():

    figure_rank_variation()


def figure_3():

    files = ["./fig/doctorate_group_colors.pdf", "./fig/lorentz_curve_group.pdf"]

    combine_and_put_labels_v2(files, result_name="./fig/figure_3.pdf")


def figure_4():

    files = ["./fig/doctorate_reigon.pdf", "./fig/lorentz_curve_region.pdf"]

    combine_and_put_labels(files, result_name="./fig/figure_4.pdf")


def combine_and_put_labels(pdf_files, result_name="./result.pdf", dir="vertical"):

    def _add_labels_to_figures(axs, labels):
        for ax, label in zip(axs, labels):
            ax.text(0.02, 0.98, label, ha='left', va='top', fontsize=12, transform=ax.transAxes)

    images = []

    for file in pdf_files:
        pages = convert_from_path(file, dpi=200)
        images.append(pages[0])

    plt.rc('font', **param_font)

    if dir == "vertical":
        fig, axs = plt.subplots(len(images), 1, figsize=(param_fig_xsize * len(images), param_fig_ysize), dpi=200, constrained_layout=True)

    else:
        fig, axs = plt.subplots(1, len(images), figsize=(param_fig_xsize, param_fig_ysize * len(images)), dpi=200, constrained_layout=True)

    for i, img in enumerate(images):
        axs[i].imshow(img)

    # _add_labels_to_figures(axs, ['a)', 'b)', 'c)'])

    for i, ax in enumerate(axs.flat):

        label = f"({chr(97 + i)})"
        ax.text(-0.02, 1.01, label, transform=ax.transAxes, fontsize=param_pannel_size, fontweight='bold', va='top')

    for ax in axs:
        ax.axis('off')

    plt.tight_layout(pad=1)
    plt.savefig(result_name, bbox_inches='tight')

    plt.clf()


def combine_and_put_labels_v2(pdf_files, result_name="./result.pdf", dir="vertical"):
    images = []

    for file in pdf_files:
        pages = convert_from_path(file, dpi=200)
        images.append(pages[0])

    plt.rc('font', **param_font)

    if dir == "vertical":
        fig, axs = plt.subplots(len(images), 1, figsize=(param_fig_xsize * len(images), param_fig_ysize), dpi=200, constrained_layout=True)
    else:
        fig, axs = plt.subplots(1, len(images), figsize=(param_fig_xsize, param_fig_ysize * len(images)), dpi=200, constrained_layout=True)

    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
        label = f"({chr(97 + i)})"
        axs[i].text(-0.02, 1.01, label, transform=axs[i].transAxes, fontsize=param_pannel_size, fontweight='bold', va='top')

    with PdfPages(result_name) as pdf:
        plt.savefig(pdf, format='pdf', bbox_inches='tight')

    plt.clf()


def figure_lorentz_curve():

    fig_path = "./fig/lorentz_curve.pdf"

    networks_global = facsimlib.processing.construct_network()
    networks_domestic = facsimlib.processing.construct_network(net_type='domestic')

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 2, figsize=(2 * param_fig_xsize, param_fig_ysize), dpi=200)

    _figure_lorentz_curve(networks_global, ax[0])
    _figure_lorentz_curve(networks_domestic, ax[1])

    for i, ax in enumerate(ax.flat):

        label = f"({chr(97 + i)})"
        ax.text(-0.2, 1.1, label, transform=ax.transAxes, fontsize=param_pannel_size, fontweight='bold', va='top')

    handles = [Patch(facecolor=networks_global["Biology"].color, label="Biology",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=networks_global["Computer Science"].color, label="Computer Science",
                     alpha=param_alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=networks_global["Physics"].color, label="Physics",
                     alpha=param_alpha, edgecolor='black', linewidth=3)]

    labels = ["Biology", "Computer Science", "Physics"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=5, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _figure_lorentz_curve(network_dict, ax):

    div_10per = [i for i in range(0, 101, 20)]

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))

    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))

    x_label = "Cumulative Ratio of Institutions (%)"
    y_label = "Cumulative Ratio of Out Degrees (%)"

    ax.set_xlabel(x_label, fontsize=param_xlabel_size)
    ax.set_ylabel(y_label, fontsize=param_ylabel_size)

    for net in network_dict.values():

        out_degrees = [net.net.out_degree(node) for node in net.net.nodes]

        (gini_coeff, x_co, y_co, base_co) = process_gini_coeff(out_degrees)

        ax.plot(x_co, y_co, color=net.color, linewidth=3, alpha=param_alpha)
        ax.scatter(div_10per, [sample_from_data(x_co, y_co, index) for index in div_10per],
                   c=net.color, s=30, clion=False, alpha=1, marker='o')

        ax.plot(base_co, base_co, color='black', linewidth=0.5)

        if net.name == "Biology" or net.name == "Domestic Biology":
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

            text_to_put = math.floor(gini_coeff * 1000) / 1000

            ax.text(centroid_x / 100, centroid_y / 100, text_to_put, transform=ax.transAxes,
                    ha='center', va='center', fontsize=param_pannel_size, fontweight='bold')


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


def figure_rank_variation():

    fig_path = "./fig/rank_variation.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.05, 0.25), fontsize=param_tick_size)

    _figure_rank_variation(network_dict["Biology"], ax[0])
    _figure_rank_variation(network_dict["Computer Science"], ax[1])
    _figure_rank_variation(network_dict["Physics"], ax[2])

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


def figure_rank_variation_random():

    fig_path = "./fig/rank_variation_random.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.05, 0.25), fontsize=param_tick_size)

    variations = _figure_rank_variation(network_dict["Biology"], ax[0])
    _figure_rank_variation_random(network_dict["Biology"], ax[0], to_compare=variations)

    variations = _figure_rank_variation(network_dict["Computer Science"], ax[1])
    _figure_rank_variation_random(network_dict["Computer Science"], ax[1], to_compare=variations)

    variations = _figure_rank_variation(network_dict["Physics"], ax[2])
    _figure_rank_variation_random(network_dict["Physics"], ax[2], to_compare=variations)

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


def figure_rank_variation_random_zscore():

    fig_path = "./fig/rank_variation_random_zscore.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    # plt.ylim(-5, 5)
    # plt.yticks(np.arange(-5, 6, 2.5), fontsize=param_tick_size)

    variations = _figure_rank_variation(network_dict["Biology"], None)
    _figure_rank_variation_random_zscore(network_dict["Biology"], ax[0], to_compare=variations, marker='o')

    variations = _figure_rank_variation(network_dict["Computer Science"], None)
    _figure_rank_variation_random_zscore(network_dict["Computer Science"], ax[1], to_compare=variations, marker='o')

    variations = _figure_rank_variation(network_dict["Physics"], None)
    _figure_rank_variation_random_zscore(network_dict["Physics"], ax[2], to_compare=variations, marker='o')

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


def figure_rank_variation_random_zscore_vs_ratio():

    fig_path = "./fig/rank_variation_random_zscore_vs_ratio.pdf"

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():
        net.set_ranks()

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200)

    # plt.ylim(-5, 5)
    # plt.yticks(np.arange(-5, 6, 2.5), fontsize=param_tick_size)

    variations = _figure_rank_variation(network_dict["Biology"], None)
    _figure_rank_variation_random_zscore_vs_ratio(network_dict["Biology"], ax[0], to_compare=variations, marker='o')

    variations = _figure_rank_variation(network_dict["Computer Science"], None)
    _figure_rank_variation_random_zscore_vs_ratio(network_dict["Computer Science"], ax[1], to_compare=variations, marker='o')

    variations = _figure_rank_variation(network_dict["Physics"], None)
    _figure_rank_variation_random_zscore_vs_ratio(network_dict["Physics"], ax[2], to_compare=variations, marker='o')

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


def _figure_rank_variation(network, ax, marker='o'):

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


def _figure_rank_variation_random(network, ax, marker='x', to_compare=None):

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


def _figure_rank_variation_random_zscore(network, ax, marker='x', to_compare=None):

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


def _figure_rank_variation_random_zscore_vs_ratio(network, ax, marker='x', to_compare=None):

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
                print(f"{network.name}: {domestic_ranks[key]}")
                
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

    # lin_co = np.arange(0, 1, 0.05)

    # ax.plot(lin_co, poly1d_func(lin_co), c='red', alpha=param_alpha)

    # sorted_data = sorted(zip(x_co, stds))
    # x_co_sorted, stds_sorted = zip(*sorted_data)

    # ax.plot(x_co_sorted, stds_sorted, marker='s', c='red', linestyle='--', alpha=param_alpha)


def figure_doctorate_group():

    fig_path = "./fig/doctorate_group.pdf"

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

    _figure_doctorate_group(networks_dict["Biology"], ax[0])
    _figure_doctorate_group(networks_dict["Computer Science"], ax[1])
    _figure_doctorate_group(networks_dict["Physics"], ax[2])

    handles1 = [Patch(facecolor=networks_dict["Biology"].color, label="Biology", alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=networks_dict["Computer Science"].color, label="Computer Science", alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=networks_dict["Physics"].color, label="Physics", alpha=param_alpha, edgecolor='black', linewidth=3)]
    
    handles2 = [Patch(edgecolor='black', hatch='*', facecolor='white'),
                Patch(edgecolor='black', hatch='O', facecolor='white'),
                Patch(edgecolor='black', hatch='+', facecolor='white'),
                Patch(edgecolor='black', hatch='-', facecolor='white'),
                Patch(edgecolor='black', hatch='/', facecolor='white')]

    labels1 = ["Biology", "Computer Science", "Physics"]
    labels2 = ["America", "Asia", "Europe", "Oceania", "South Korea"]

    fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
    fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.045), ncol=5, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _figure_doctorate_group(network, ax, use_hatches=True):
    
    num_group = 10

    total_count = {}

    kr_count = {}
    asia_count = {}
    america_count = {}
    europe_count = {}
    oceania_count = {}

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
        asia_count[rank] = 0
        america_count[rank] = 0
        europe_count[rank] = 0
        oceania_count[rank] = 0

        for src, dst, data in edges_in:

            if (ns_korea.hit(network.net.nodes[src])):
                kr_count[rank] += 1

            elif ((ns_asia.hit(network.net.nodes[src]))):
                asia_count[rank] += 1

            elif ((ns_america.hit(network.net.nodes[src]))):
                america_count[rank] += 1

            elif ((ns_europe.hit(network.net.nodes[src]))):
                europe_count[rank] += 1

            elif ((ns_oceania.hit(network.net.nodes[src]))):
                oceania_count[rank] += 1

    ranks_sorted = [key for key, _ in sorted(total_count.items())]
    kr_count_sorted = [value for _, value in sorted(kr_count.items())]
    asia_count_sorted = [value for _, value in sorted(asia_count.items())]
    america_count_sorted = [value for _, value in sorted(america_count.items())]
    europe_count_sorted = [value for _, value in sorted(europe_count.items())]
    oceania_count_sorted = [value for _, value in sorted(oceania_count.items())]

    grousize = np.floor(len(ranks_sorted) / num_group)

    if (grousize == 1):

        x_co = ranks_sorted
        y_co_kr = kr_count_sorted
        y_co_asia = asia_count_sorted
        y_co_america = america_count_sorted
        y_co_europe = europe_count_sorted
        y_co_oceania = oceania_count_sorted

    else:
        
        x_co = []

        y_co_kr = []
        y_co_asia = []
        y_co_america = []
        y_co_europe = []
        y_co_oceania = []
        
        index = 0
        grouid = 1

        while index < num_group * grousize:

            elements_kr = []
            elements_asia = []
            elements_america = []
            elements_europe = []
            elements_oceania = []

            while (grousize > len(elements_kr)):

                elements_kr.append(kr_count_sorted[index])
                elements_asia.append(asia_count_sorted[index])
                elements_america.append(america_count_sorted[index])
                elements_europe.append(europe_count_sorted[index])
                elements_oceania.append(oceania_count_sorted[index])

                index += 1

                if (index >= len(ranks_sorted)):
                    break

            x_co.append(grouid)

            y_co_kr.append(sum(elements_kr))
            y_co_asia.append(sum(elements_asia))
            y_co_america.append(sum(elements_america))
            y_co_europe.append(sum(elements_europe))
            y_co_oceania.append(sum(elements_oceania))

            grouid += 1

        while index < len(ranks_sorted):

            y_co_kr[-1] += kr_count_sorted[index]
            y_co_asia[-1] += asia_count_sorted[index]
            y_co_america[-1] += america_count_sorted[index]
            y_co_europe[-1] += europe_count_sorted[index]
            y_co_oceania[-1] += oceania_count_sorted[index]

            index += 1

    y_co_kr_norm = [y_co_kr[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                    if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]
    
    y_co_asia_norm = [y_co_asia[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                      if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]
    
    y_co_america_norm = [y_co_america[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                         if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]
    
    y_co_europe_norm = [y_co_europe[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                        if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]
    
    y_co_oceania_norm = [y_co_oceania[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                         if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]

    y_co_kr = y_co_kr_norm
    y_co_asia = y_co_asia_norm
    y_co_america = y_co_america_norm
    y_co_europe = y_co_europe_norm
    y_co_oceania = y_co_oceania_norm

    ax.tick_params(axis='both', which='major', labelsize=param_tick_size)

    ax.set_xlim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))

    if use_hatches:

        ax.barh(x_co, y_co_america, color=network.color, label='America',
                hatch='*', alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_asia, color=network.color, left=y_co_america, label='Asia',
                hatch='O', alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_europe, color=network.color, left=[y_co_asia[i] + y_co_america[i] for i in range(len(y_co_asia))], label='Europe',
                hatch='+', alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_oceania, color=network.color, left=[y_co_asia[i] + y_co_america[i] + y_co_europe[i] for i in range(len(y_co_asia))], label='Oceania',
                hatch='-', alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_kr, color=network.color, left=[y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i] for i in range(len(y_co_asia))], label='South Korea',
                hatch='/', alpha=param_alpha, edgecolor='black', linewidth=2)
        
        return None
        
    else:

        palette = split_color_by(network.color)

        ax.barh(x_co, y_co_america, color=palette[0], label='America',
                alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_asia, color=palette[1], left=y_co_america, label='Asia',
                alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_europe, color=palette[2], left=[y_co_asia[i] + y_co_america[i] for i in range(len(y_co_asia))], label='Europe',
                alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_oceania, color=palette[3], left=[y_co_asia[i] + y_co_america[i] + y_co_europe[i] for i in range(len(y_co_asia))], label='Oceania',
                alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_kr, color=palette[4], left=[y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i] for i in range(len(y_co_asia))], label='South Korea',
                alpha=param_alpha, edgecolor='black', linewidth=2)

        return palette
    

def figure_doctorate_group_using_colors():

    fig_path = "./fig/doctorate_group_colors.pdf"

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

    palette_bio = _figure_doctorate_group(networks_dict["Biology"], ax[0], use_hatches=False)
    palette_cs = _figure_doctorate_group(networks_dict["Computer Science"], ax[1], use_hatches=False)
    palette_phy = _figure_doctorate_group(networks_dict["Physics"], ax[2], use_hatches=False)

    handles_bio = [Patch(facecolor=palette_bio[i], alpha=param_alpha, edgecolor='black', linewidth=3) for i in range(len(palette_bio))]
    handles_cs = [Patch(facecolor=palette_cs[i], alpha=param_alpha, edgecolor='black', linewidth=3) for i in range(len(palette_cs))]
    handles_phy = [Patch(facecolor=palette_phy[i], alpha=param_alpha, edgecolor='black', linewidth=3) for i in range(len(palette_phy))]
    
    labels_root = ["America", "Asia", "Europe", "Oceania", "South Korea"]

    labels_bio = [f"Biology ({root})" for root in labels_root]
    labels_cs = [f"Computer Science ({root})" for root in labels_root]
    labels_phy = [f"Physics ({root})" for root in labels_root]

    handles = [h_field[i] for i in range(5) for h_field in [handles_bio, handles_cs, handles_phy]]
    labels = [l_field[i] for i in range(5) for l_field in [labels_bio, labels_cs, labels_phy]]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def figure_lorentz_curve_group():
    
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

    _figure_lorentz_curve_group(networks_dict["Biology"], ax[0])
    _figure_lorentz_curve_group(networks_dict["Computer Science"], ax[1])
    _figure_lorentz_curve_group(networks_dict["Physics"], ax[2])

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


def _figure_lorentz_curve_group(network, ax):

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
                   c=network.color, s=30, clion=False, alpha=1, marker='o')

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
            

def figure_doctorate_region():
    
    fig_path = "./fig/doctorate_region.pdf"

    networks_dict = facsimlib.processing.construct_network()
    
    for net in networks_dict.values():
        net.copy_ranks_from(net.domestic.set_ranks())

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200, sharey=True)
    plt.gca().invert_yaxis()

    x_label = "Number of Assistant Professors (Normalized)"

    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    _figure_doctorate_region(networks_dict["Biology"], ax[0])
    _figure_doctorate_region(networks_dict["Computer Science"], ax[1])
    _figure_doctorate_region(networks_dict["Physics"], ax[2])

    handles1 = [Patch(facecolor=networks_dict["Biology"].color, label="Biology", alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=networks_dict["Computer Science"].color, label="Computer Science", alpha=param_alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=networks_dict["Physics"].color, label="Physics", alpha=param_alpha, edgecolor='black', linewidth=3)]
    
    handles2 = [Patch(edgecolor='black', hatch='*', facecolor='white'),
                Patch(edgecolor='black', hatch='O', facecolor='white'),
                Patch(edgecolor='black', hatch='+', facecolor='white'),
                Patch(edgecolor='black', hatch='-', facecolor='white'),
                Patch(edgecolor='black', hatch='/', facecolor='white')]

    labels1 = ["Biology", "Computer Science", "Physics"]
    labels2 = ["America", "Asia", "Europe", "Oceania", "South Korea"]

    fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
    fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.045), ncol=5, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def figure_doctorate_region_using_colors():
    
    fig_path = "./fig/doctorate_region_colors.pdf"

    networks_dict = facsimlib.processing.construct_network()
    
    for net in networks_dict.values():
        net.copy_ranks_from(net.domestic.set_ranks())

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(1, 3, figsize=(3 * param_fig_xsize, param_fig_ysize), dpi=200, sharey=True)
    plt.gca().invert_yaxis()

    x_label = "Number of Assistant Professors (Normalized)"

    ax[1].set_xlabel(x_label, fontsize=param_xlabel_size)

    _figure_doctorate_region(networks_dict["Biology"], ax[0], using_hatches=False)
    _figure_doctorate_region(networks_dict["Computer Science"], ax[1], using_hatches=False)
    _figure_doctorate_region(networks_dict["Physics"], ax[2], using_hatches=False)

    # handles1 = [Patch(facecolor=networks_dict["Biology"].color, label="Biology", alpha=param_alpha, edgecolor='black', linewidth=3),
    #             Patch(facecolor=networks_dict["Computer Science"].color, label="Computer Science", alpha=param_alpha, edgecolor='black', linewidth=3),
    #             Patch(facecolor=networks_dict["Physics"].color, label="Physics", alpha=param_alpha, edgecolor='black', linewidth=3)]
    
    # handles2 = [Patch(edgecolor='black', hatch='*', facecolor='white'),
    #             Patch(edgecolor='black', hatch='O', facecolor='white'),
    #             Patch(edgecolor='black', hatch='+', facecolor='white'),
    #             Patch(edgecolor='black', hatch='-', facecolor='white'),
    #             Patch(edgecolor='black', hatch='/', facecolor='white')]

    # labels1 = ["Biology", "Computer Science", "Physics"]
    # labels2 = ["America", "Asia", "Europe", "Oceania", "South Korea"]

    # fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
    # fig.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.045), ncol=5, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def _figure_doctorate_region(network, ax, using_hatches=True):

    ns_list = [NS('region', area_seoul, 'in', label="Seoul"),
               NS('region', area_capital, 'in', label="Capital Area"),
               NS('region', area_metro, 'in', label="Metropolitan\nCities"),
               NS('region', area_others, 'in', label="Others"),
               NS('name', inst_ists, 'in', label="-IST")]
    
    total_count = {}

    kr_count = {}
    asia_count = {}
    america_count = {}
    europe_count = {}
    oceania_count = {}

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")
    ns_asia = NS('country_code', con_asia_without_kr, 'in', label="Asia")
    ns_america = NS('country_code', con_america, 'in', label="America")
    ns_europe = NS('country_code', con_europe, 'in', label="Europe")
    ns_oceania = NS('country_code', con_ocenaia, 'in', label="Europe")

    for ns in ns_list:

        total_count[ns.label] = 0
        kr_count[ns.label] = 0
        asia_count[ns.label] = 0
        america_count[ns.label] = 0
        europe_count[ns.label] = 0
        oceania_count[ns.label] = 0

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
                    asia_count[ns_repr] += 1

            elif ((ns_america.hit(network.net.nodes[src]))):
                for ns_repr in ns_list_hit:
                    america_count[ns_repr] += 1

            elif ((ns_europe.hit(network.net.nodes[src]))):
                for ns_repr in ns_list_hit:
                    europe_count[ns_repr] += 1

            elif ((ns_oceania.hit(network.net.nodes[src]))):
                for ns_repr in ns_list_hit:
                    oceania_count[ns_repr] += 1

    total_count_sorted = [value for _, value in sorted(total_count.items())]

    x_co = [ns.label if ns.label is not None else ns.__repr__() for ns in ns_list]

    y_co_kr = [value for _, value in sorted(kr_count.items())]
    y_co_asia = [value for _, value in sorted(asia_count.items())]
    y_co_america = [value for _, value in sorted(america_count.items())]
    y_co_europe = [value for _, value in sorted(europe_count.items())]
    y_co_oceania = [value for _, value in sorted(oceania_count.items())]
            
    y_co_kr_norm = [y_co_kr[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]
    y_co_asia_norm = [y_co_asia[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]
    y_co_america_norm = [y_co_america[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]
    y_co_europe_norm = [y_co_europe[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]
    y_co_oceania_norm = [y_co_oceania[i] / (total_count_sorted[i]) if total_count_sorted[i] != 0 else 0 for i in range(len(x_co))]

    y_co_kr = y_co_kr_norm
    y_co_asia = y_co_asia_norm
    y_co_america = y_co_america_norm
    y_co_europe = y_co_europe_norm
    y_co_oceania = y_co_oceania_norm

    ax.set_xlim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))

    if using_hatches:
    
        ax.barh(x_co, y_co_america, color=network.color, label='America',
                hatch='*', alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_asia, color=network.color, label='Asia', left=y_co_america, 
                hatch='O', alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_europe, color=network.color, left=[y_co_asia[i] + y_co_america[i] for i in range(len(y_co_asia))], label='Europe',
                hatch='+', alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_oceania, color=network.color, left=[y_co_asia[i] + y_co_america[i] + y_co_europe[i] for i in range(len(y_co_asia))], label='Oceania',
                hatch='-', alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_kr, color=network.color, left=[y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i] for i in range(len(y_co_asia))], label='South Korea',
                hatch='/', alpha=param_alpha, edgecolor='black', linewidth=2)

    else:

        palette = split_color_by(network.color)

        ax.barh(x_co, y_co_america, color=palette[0], label='America',
                alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_asia, color=palette[1], label='Asia', left=y_co_america,
                alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_europe, color=palette[2], left=[y_co_asia[i] + y_co_america[i] for i in range(len(y_co_asia))], label='Europe',
                alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_oceania, color=palette[3], left=[y_co_asia[i] + y_co_america[i] + y_co_europe[i] for i in range(len(y_co_asia))], label='Oceania',
                alpha=param_alpha, edgecolor='black', linewidth=2)
        ax.barh(x_co, y_co_kr, color=palette[4], left=[y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i] for i in range(len(y_co_asia))], label='South Korea',
                alpha=param_alpha, edgecolor='black', linewidth=2)


def figure_lorentz_curve_region():
    
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

    _figure_lorentz_curve_region(networks_dict["Biology"], ax[0])
    _figure_lorentz_curve_region(networks_dict["Computer Science"], ax[1])
    _figure_lorentz_curve_region(networks_dict["Physics"], ax[2])

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


def _figure_lorentz_curve_region(network, ax):
    
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
                   c=network.color, s=30, clion=False, alpha=1, marker='o')

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


def figure_hires():
    
    fig_path = "./fig/hires.pdf"

    networks_global = facsimlib.processing.construct_network()
    networks_domestic = facsimlib.processing.construct_network(net_type='domestic')

    plt.rc('font', **param_font)

    fig, ax = plt.subplots(2, 2, figsize=(2 * param_fig_xsize, 2 * param_fig_ysize), dpi=200)

    _figure_hires_distro(networks_global, ax[0, 0])
    _figure_hires_z(networks_global, ax[0, 1])

    _figure_hires_distro(networks_domestic, ax[1, 0])
    _figure_hires_z(networks_domestic, ax[1, 1])

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

    plt.tight_layout(pad=1)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def figure_hires_using_colors():
    
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


def _figure_hires_distro(network_dict, ax, using_hatches=True):

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

        if using_hatches:

            ax.bar(name_to_put, down_hires[net.name], color=net.color, hatch='x', alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(name_to_put, self_hires[net.name], color=net.color, bottom=down_hires[net.name], hatch='o',
                   alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(name_to_put, uhires[net.name], color=net.color, bottom=down_hires[net.name] + self_hires[net.name],
                   hatch='-', alpha=param_alpha, edgecolor='black', linewidth=2)
            
        else:

            palette = split_color_by(net.color, 3)

            ax.bar(name_to_put, down_hires[net.name], color=palette[0], alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(name_to_put, self_hires[net.name], color=palette[1], bottom=down_hires[net.name],
                   alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(name_to_put, uhires[net.name], color=palette[2], bottom=down_hires[net.name] + self_hires[net.name],
                   alpha=param_alpha, edgecolor='black', linewidth=2)
            

def _figure_hires_z(network_dict, ax, using_hatches=True):

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

    bar_width = 0.25
    x_pos = 0

    for net in network_dict.values():

        x_pos_1 = x_pos
        x_pos_2 = x_pos + bar_width
        x_pos_3 = x_pos + 2 * bar_width

        if using_hatches:

            ax.bar(x_pos_1, uhires[net.name], width=bar_width, color=net.color,
                   hatch='-', alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(x_pos_2, self_hires[net.name], width=bar_width, color=net.color,
                   hatch='o', alpha=param_alpha, edgecolor='black', linewidth=2)
            ax.bar(x_pos_3, down_hires[net.name], width=bar_width, color=net.color,
                   hatch='x', alpha=param_alpha, edgecolor='black', linewidth=2)
            
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


if __name__ == '__main__':
    
    figure_rank_variation_random_zscore_vs_ratio()