import facsimlib.processing
from facsimlib.academia import Field
import facsimlib.math


def construct_network():

    prep_list = facsimlib.processing.preprocess("data")
    network_dict = {}

    for field, df_list in prep_list:

        network_dict[field] = Field(name=field)
        facsimlib.processing.process_file(df_list, network_dict[field])

    return network_dict


if (__name__ == "__main__"):

    networks = construct_network()

    net_bio = networks["Biology"].closed
    net_cs = networks["Computer Science"].closed
    net_phy = networks["Physics"].closed

    for net in [net_bio, net_cs, net_phy]:
        net.set_ranks()

    rs_bc = facsimlib.math.spearman_rank_correlation(net_bio, net_cs)
    rs_bp = facsimlib.math.spearman_rank_correlation(net_bio, net_phy)
    rs_cp = facsimlib.math.spearman_rank_correlation(net_cs, net_phy)

    rp_bc = facsimlib.math.pearson_rank_correlation(net_bio, net_cs)
    rp_bp = facsimlib.math.pearson_rank_correlation(net_bio, net_phy)
    rp_cp = facsimlib.math.pearson_rank_correlation(net_cs, net_phy)

    print('=' * 10 + "Spearman Correlations" + '=' * 10)
    print(f"Biology-Computer Science: {rs_bc}")
    print(f"Computer Science-Physics: {rs_bp}")
    print(f"Physics-Biology: {rs_bp}")

    print('=' * 10 + "Pearson Correlations" + '=' * 10)
    print(f"Biology-Computer Science: {rp_bc}")
    print(f"Computer Science-Physics: {rp_bp}")
    print(f"Physics-Biology: {rp_bp}")
    
