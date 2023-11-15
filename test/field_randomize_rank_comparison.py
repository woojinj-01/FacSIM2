import facsimlib.processing
import facsimlib.plot

trial = 500


if (__name__ == "__main__"):

    networks = facsimlib.processing.construct_network()

    for net_source in networks.values():

        net = net_source.closed

        net_rand = net.randomize(trial)

        net.set_ranks()

        for net_r in net_rand:

            net_r.set_ranks()

        facsimlib.plot.plot_rank_comparison_multiple(net, net_rand)

    