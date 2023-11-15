import facsimlib.processing
from facsimlib.academia import Field
from facsimlib.text import normalize_inst_name


def construct_network():

    prep_list = facsimlib.processing.preprocess("data")
    network_dict = {}

    for field, df_list in prep_list:

        network_dict[field] = Field(name=field)
        facsimlib.processing.process_file(df_list, network_dict[field])

    return network_dict


if (__name__ == "__main__"):

    kaist = normalize_inst_name('Korea Advanced Institute of Science and Technology,Daejeon,KR')
    snu = normalize_inst_name('Seoul National University,Seoul,KR')

    networks = construct_network()

    net_bio = networks["Computer Science"]

    # net_bio_rand = net_bio.randomize(10)

    print(net_bio.name)

    print("In degrees")
    print(net_bio.net.in_degree([kaist]))
    print("Out degrees")
    print(net_bio.net.out_degree([snu]))

    # for net in net_bio_rand:

    #     print(net.name)

    #     print(net.net.in_degree(['Korea Advanced Institute of Science and Technology,Daejeon,KR']))
    #     print(net.net.out_degree(['Korea Advanced Institute of Science and Technology,Daejeon,KR']))