from facsimlib.processing import construct_network
import facsimlib.math


if (__name__ == "__main__"):

    networks = construct_network()

    net_bio = networks["Biology"]
    net_bio_c = net_bio.closed

    net_cs = networks["Computer Science"]
    net_cs_c = net_cs.closed

    net_phy = networks["Physics"]
    net_phy_c = net_phy.closed

    for net in [net_bio, net_bio_c, net_cs, net_cs_c, net_phy, net_phy_c]:
        net.set_ranks()

    rs_bio = facsimlib.math.spearman_rank_correlation(net_bio, net_bio_c)
    rs_cs = facsimlib.math.spearman_rank_correlation(net_cs, net_cs_c)
    rs_phy = facsimlib.math.spearman_rank_correlation(net_phy, net_phy_c)

    rp_bio = facsimlib.math.pearson_rank_correlation(net_bio, net_bio_c)
    rp_cs = facsimlib.math.pearson_rank_correlation(net_cs, net_cs_c)
    rp_phy = facsimlib.math.pearson_rank_correlation(net_phy, net_phy_c)

    print('=' * 10 + "Spearman Correlations" + '=' * 10)
    print(f"Biology-Computer Science: {rs_bio}")
    print(f"Computer Science-Physics: {rs_cs}")
    print(f"Physics-Biology: {rs_phy}")

    print('=' * 10 + "Pearson Correlations" + '=' * 10)
    print(f"Biology-Computer Science: {rp_bio}")
    print(f"Computer Science-Physics: {rp_cs}")
    print(f"Physics-Biology: {rp_phy}")
    
