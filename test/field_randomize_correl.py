import numpy as np

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

    net_source = networks["Computer Science"].closed

    net_source_rand = net_source.randomize(500)

    net_source.set_ranks()

    correls = []

    trial = 1

    for net in net_source_rand:

        net.set_ranks()

        correlation = facsimlib.math.spearman_rank_correlation(net_source, net)

        correls.append(correlation)

        print(f"Trial: {trial}")
        print(f"Correlation: {correlation}\n")

        trial += 1

    print("=" * 10 + f"For {trial - 1} Trials" + "=" * 10)

    print(f"Mean {round(np.mean(correls), 4)}, Std.Dev {round(np.std(correls), 4)}, Max {round(max(correls), 4)}, Min {round(min(correls), 4)}")
    
