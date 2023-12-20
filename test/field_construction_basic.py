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

    kaist = 'Korea Advanced Institute of Science and Technology,Daejeon,KR'
    snu = 'Seoul National University,Seoul,KR'
    ewha = "ewha womans university,seoul,KR"

    networks = construct_network()

    for net in networks.values():

        net_closed = net.closed

        net_closed.set_ranks()
        net_closed.export_ranks()

    # print(net_bio)
    
    # for edge in net_bio.net.edges(data=True):
    #     print(edge)

    # print(net_bio.move(snu, kaist))
    # print(net_bio.net.out_edges(snu))
    # print(net_bio.move(snu, ewha)[1]['current_rank'])
    # print(len(net_bio.move(snu, ewha)))
