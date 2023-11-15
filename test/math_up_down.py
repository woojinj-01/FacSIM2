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

    for network in networks.values():

        network.set_ranks()

        print(f"{network.name}: {facsimlib.math.up_down_hires(network)}")
