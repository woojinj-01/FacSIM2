import facsimlib.processing
from facsimlib.academia import Field


if (__name__ == "__main__"):

    prep_list = facsimlib.processing.preprocess("data")
    network_dict = {}

    for field, df_list in prep_list:

        network_dict[field] = Field(field)
        facsimlib.processing.process_file(df_list, network_dict[field])

        print(network_dict[field].net.nodes['Korea Advanced Institute of Science and Technology,Daejeon,KR']['inst_id'])






