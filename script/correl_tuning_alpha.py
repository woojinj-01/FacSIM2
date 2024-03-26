import sys
sys.path.append('/Users/woojinj/Desktop/FacSIM2')

import numpy as np
import matplotlib.pyplot as plt

from facsimlib.processing import construct_network
from facsimlib.math import spearman_rank_correlation, pearson_rank_correlation
from facsimlib.plot.general import param_alpha


def set_ranks_all(nets, alpha):

    for net in nets.values():

        net.reset_ranks()
        net.set_ranks(add_one=False, alpha=alpha)


def tune_alpha_and_compare(alpha_list):
    
    nets = construct_network()
    nets_to_comp = construct_network()

    for net in nets.values():
        net.set_ranks(add_one=True)

    def cal_correl():
        
        for name in nets.keys():

            correl_sp = spearman_rank_correlation(nets[name], nets_to_comp[name])
            correl_pe = pearson_rank_correlation(nets[name], nets_to_comp[name])

            print(f">>> {name}")
            print(f"Spearman: {correl_sp}")
            print(f"Pearson: {correl_pe}")

    for alpha in alpha_list:

        set_ranks_all(nets_to_comp, alpha)
        
        print(f"=== Alpha: {alpha} ===")
        cal_correl()


def tune_alpha_and_plot(alpha_list, net_type):

    nets = construct_network(net_type=net_type)
    nets_to_comp = construct_network(net_type=net_type)

    min_alpha = min(alpha_list)
    max_alpha = max(alpha_list)

    for net in nets.values():
        net.set_ranks(add_one=True)

    for name in nets.keys():

        correls = []

        for alpha in alpha_list:

            set_ranks_all(nets_to_comp, alpha)

            correl_sp = spearman_rank_correlation(nets[name], nets_to_comp[name])

            correls.append(correl_sp)

        plt.plot(alpha_list, correls, label=nets[name].name, c=nets[name].color, alpha=param_alpha)

    plt.xlim(min_alpha, max_alpha)
    plt.hlines(1, min_alpha, max_alpha, colors='black', linestyles='dashed', linewidth=1)

    plt.xlabel("Alpha")
    plt.ylabel("Correlation (vs. Adding 1 scheme)")

    plt.legend(loc="lower right")
    plt.savefig(f"./fig/alpha_effect_{net_type}.pdf")

    plt.clf()


if __name__ == "__main__":

    alpha_list = np.arange(0, 15.01, 0.05)

    tune_alpha_and_plot(alpha_list, 'global')
    tune_alpha_and_plot(alpha_list, 'domestic')

        





    