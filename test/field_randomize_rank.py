import numpy as np

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

    inst1 = 'Seoul National University,Seoul,KR'
    inst2 = 'Pohang University of Science and Technology,Pohang,KR'

    networks = construct_network()

    net_source = networks["Computer Science"]

    net_source_rand = net_source.randomize(500)

    net_source.set_ranks()

    inst1_ranks = []
    inst2_ranks = []

    trial = 1

    print(f"Inst 1 name: {inst1}")
    print(f"Inst 2 name: {inst2}\n")

    for net in net_source_rand:

        net.set_ranks()

        inst1_ranks.append(net.inst(inst1)['rank'])
        inst2_ranks.append(net.inst(inst2)['rank'])

        trial += 1

    print("=" * 10 + f"For {trial - 1} Trials" + "=" * 10)

    print(f"Inst 1 : Reference {net_source.inst(inst1)['rank']}, Mean {np.mean(inst1_ranks)}, Std.Dev {np.std(inst1_ranks)}, Max {max(inst1_ranks)}, Min {min(inst1_ranks)}")
    print(f"Inst 2: Reference {net_source.inst(inst2)['rank']}, Mean {np.mean(inst2_ranks)}, Std.Dev {np.std(inst2_ranks)}, Max {max(inst2_ranks)}, Min {min(inst2_ranks)}")
