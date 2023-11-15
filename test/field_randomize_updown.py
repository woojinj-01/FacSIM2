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

    networks_closed = {net.name: net.closed for net in networks.values()}

    trial = 500

    for net in networks_closed.values():

        net_random = net.randomize(trial)

        net.set_ranks()

        print(f"Field: {net.name}")

        print(f"Original: {facsimlib.math.up_down_hires(net)}")

        up_hires = []
        self_hires = []
        down_hires = []

        for net_rand in net_random:

            net_rand.set_ranks()

            stat = facsimlib.math.up_down_hires(net_rand)

            up_hires.append(stat[0])
            self_hires.append(stat[1])
            down_hires.append(stat[2])

        up_avg = sum(up_hires) / len(up_hires)
        se_avg = sum(self_hires) / len(self_hires)
        down_avg = sum(down_hires) / len(down_hires)

        print(f"Random (For {trial} Trials): ({up_avg}, {se_avg}, {down_avg})")
