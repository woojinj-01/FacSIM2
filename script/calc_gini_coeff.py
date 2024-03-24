from facsimlib.processing import construct_network
from facsimlib.plot.general import process_gini_coeff

if __name__ == "__main__":

    nets_global = construct_network()
    nets_domestic = construct_network(net_type="domestic")

    for nets in [nets_global, nets_domestic]:

        for net in nets.values():

            out_degrees = [net.net.out_degree(node) for node in net.net.nodes]
            
            gini, _, _, _ = process_gini_coeff(out_degrees)

            print(f"{net.name}: {gini}")