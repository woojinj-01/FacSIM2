import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import math

import facsimlib.processing
import facsimlib.math
from facsimlib.plot.params import *


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

