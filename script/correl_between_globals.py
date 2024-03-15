from facsimlib.processing import construct_network
from facsimlib.math import spearman_rank_correlation, pearson_rank_correlation


if (__name__ == "__main__"):

    networks = construct_network()

    net_bio = networks["Biology"]
    net_bio_d = net_bio.domestic

    net_cs = networks["Computer Science"]
    net_cs_d = net_cs.domestic

    net_phy = networks["Physics"]
    net_phy_d = net_phy.domestic

    for net in [net_bio, net_bio_d, net_cs, net_cs_d, net_phy, net_phy_d]:
        net.set_ranks()

    rs_bio = spearman_rank_correlation(net_bio, net_bio_d)
    rs_cs = spearman_rank_correlation(net_cs, net_cs_d)
    rs_phy = spearman_rank_correlation(net_phy, net_phy_d)

    rp_bio = pearson_rank_correlation(net_bio, net_bio_d)
    rp_cs = pearson_rank_correlation(net_cs, net_cs_d)
    rp_phy = pearson_rank_correlation(net_phy, net_phy_d)

    print('=' * 10 + "Spearman Correlations" + '=' * 10)
    print(f"Biology-Computer Science: {rs_bio}")
    print(f"Computer Science-Physics: {rs_cs}")
    print(f"Physics-Biology: {rs_phy}")

    print('=' * 10 + "Pearson Correlations" + '=' * 10)
    print(f"Biology-Computer Science: {rp_bio}")
    print(f"Computer Science-Physics: {rp_cs}")
    print(f"Physics-Biology: {rp_phy}")
    
