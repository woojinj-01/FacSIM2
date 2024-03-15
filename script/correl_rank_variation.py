from facsimlib.processing import construct_network
from facsimlib.academia import Field
from facsimlib.academia import NodeSelect as NS
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


def examine_correl(net: Field):

    net_d = net.domestic

    net.set_ranks()
    net_d.set_ranks()

    ranks = net.ranks(inverse=True, normalized=True)
    ranks_d = net_d.ranks(inverse=True, normalized=True)

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")

    rank_var_list = []
    interphd_ratio_list = []
    total_deg_list = []

    def _plot_variables(target_name, target_list, target2_list):

        plt.xlabel("Rank varation")
        plt.ylabel(target_name)

        # plt.scatter(rank_var_list, target_list, s=20, c="#0584F2")

        for i in range(len(rank_var_list)):

            var = rank_var_list[i]
            target = target_list[i]

            plt.scatter([var], [target], s=target2_list[i], c="#0584F2")

        plt.show()
        plt.clf()

    def _print_correl(target_name, target_list):

        print(rank_var_list)
        print(target_list)

        correl_spearman, _ = spearmanr(rank_var_list, target_list)
        correl_pearson, _ = pearsonr(rank_var_list, target_list)

        print(f"=== Against {target_name} ===")
        print(f"Spearman correlation: {round(correl_spearman, 4)}")
        print(f"Pearson correlation: {round(correl_pearson, 4)}")

    def _get_rank_variation(inst_name):

        if any(inst_name not in rank_dict for rank_dict in [ranks, ranks_d]):
            return None

        rank_g = ranks[inst_name]
        rank_d = ranks_d[inst_name]

        if any(rank is None for rank in [rank_g, rank_d]):
            return None
        
        return rank_d - rank_g
    
    def _get_interphd_ratio(inst_name):

        edges_in = net.net.in_edges(inst_name, data=True)

        kr_count = 0
        total_count = 0

        for src, _, _ in edges_in:
            
            total_count += 1

            if (ns_korea.hit(net.net.nodes[src])):
                kr_count += 1

        if total_count == 0:
            return None
        
        ratio = (total_count - kr_count) / total_count

        if ratio == 1 or ratio == 0:
            print(inst_name, ranks_d[inst_name])

        return ratio
    
    def _get_total_deg(inst_name):
        return len(net.net.in_edges(inst_name, data=True))
    
    for inst_name in ranks_d.keys():
        
        rank_var = _get_rank_variation(inst_name)
        interphd_ratio = _get_interphd_ratio(inst_name)
        total_deg = _get_total_deg(inst_name)

        if all(param is not None for param in [rank_var, interphd_ratio]):

            rank_var_list.append(rank_var)
            interphd_ratio_list.append(interphd_ratio)
            total_deg_list.append(total_deg)

    print(f"Network: {net.name}")

    _plot_variables("InterPhD ratio", interphd_ratio_list, total_deg_list)
    _print_correl("InterPhD ratio", interphd_ratio_list)


def main():

    nets = construct_network()
    examine_correl(nets["Computer Science"])


if __name__ == "__main__":
    main()

    

    
        


