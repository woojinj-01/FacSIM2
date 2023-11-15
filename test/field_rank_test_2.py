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

    net_bio = networks["Biology"]

    net_bio.set_ranks()

    kaist = net_bio.inst('korea advanced institute of science and technology,daejeon,KR')

    print(kaist['rank'])

    print("=" * 20)

    bio_closed = net_bio.closed

    bio_closed.set_ranks()

    print("=" * 20)

    bio_closed.randomize()[0].set_ranks()

    net_bio.reset_ranks()

    net_bio.copy_ranks_from(bio_closed)

