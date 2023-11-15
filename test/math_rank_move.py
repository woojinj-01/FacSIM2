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

    kaist = 'Korea Advanced Institute of Science and Technology,Daejeon,KR'
    snu = 'Seoul National University,Seoul,KR'
    ewha = "ewha womans university,seoul,KR"

    networks = construct_network()

    net_bio = networks["Biology"].set_ranks()

    print(facsimlib.math.average_relative_rank_move(net_bio))