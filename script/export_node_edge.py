from facsimlib.processing import construct_network

if __name__ == "__main__":

    nets = construct_network()

    for net in nets.values():

        net_c = net.domestic

        net.set_ranks()

        # net.export_node_list()
        # net.export_edge_list()
        net.export_ranks(advanced=True)

        net_c.set_ranks()

        # net_c.export_node_list()
        # net_c.export_edge_list()
        net_c.export_ranks(advanced=True)

