import facsimlib.processing
from facsimlib.academia import Field


def construct_network():

    prep_list = facsimlib.processing.preprocess("data")
    network_dict = {}

    for field, df_list in prep_list:

        network_dict[field] = Field(name=field)
        facsimlib.processing.process_file(df_list, network_dict[field])

    return network_dict


if (__name__ == "__main__"):

    networks = construct_network()

    net_cs = networks["Computer Science"]
    net_cs_closed = net_cs.closed

    net_cs_rand = net_cs.random
    net_cs_closed_rand = net_cs_closed.random

    net_cs.set_ranks().export_ranks()
    net_cs_rand.set_ranks().export_ranks()

    net_cs_closed.set_ranks().export_ranks()
    net_cs_closed_rand.set_ranks().export_ranks()

