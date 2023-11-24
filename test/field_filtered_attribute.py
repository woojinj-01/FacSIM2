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

    kaist = 'korea advanced institute of science and technology,daejeon,KR'
    snu = 'seoul national university,seoul,KR'
    ug = 'university of georgia,athens,US'

    networks = construct_network()

    net_bio: Field = networks["Biology"]

    net_bio_filt = net_bio.filter('country_code', ['KR', 'US'], 'in')

    print(net_bio_filt)

    # print(net_bio_filt.net.nodes[kaist])
    
    for edge in net_bio_filt.net.edges(data=True):
        print(edge)