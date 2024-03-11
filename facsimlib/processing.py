import os
import re
import pandas as pd

from facsimlib.text import are_similar_texts, normalize_inst_name
from facsimlib.academia import Institution as Inst
from facsimlib.academia import Move, Field
from facsimlib.math import calc_sparsity

cache_file_pattern = re.compile(r'^~\$')
masked_file_pattern = re.compile(r'^X_')
xlsx_file_pattern = re.compile(r'\.xlsx$')
underscore_pattern = re.compile(r'_.*_')

src_column_name = "Degree"
src_target_name = "PhD"
src_offset = 2

dst_column_name = "Job"
dst_target_name = "Assistant Professor"
dst_offset = 1


# iterator for each file (since column arrangement varies among files)
class RowIterator:

    def __init__(self, column_names: pd.core.indexes.base.Index) -> None:

        self.column_names: pd.core.indexes.base.Index = column_names
        self.index: int = 0
        self.target = None
        self.approx = 0

    def __iter__(self):
        return self
    
    def __next__(self):

        if (0 == self.approx):
            while (self.target != self.column_names[self.index].split('.')[0]):
                self.index += 1

                if (self.index >= len(self.column_names)):
                    raise StopIteration
                               
        elif (1 == self.approx):
            while (are_similar_texts(self.target, self.column_names[self.index].split('.')[0]) is None):
                self.index += 1

                if (self.index >= len(self.column_names)):
                    raise StopIteration
                
        else:
            raise StopIteration

        self.index += 1

        return self.index - 1
    
    def reset_index(self):
        self.index = 0

    def set_index_to(self, index_to_set):
        self.index = index_to_set

    def enable_approx(self):
        self.approx = 1

    def disable_approx(self):
        self.approx = 0
    
    def change_target_to(self, column_to_find: str):
        self.target: str = column_to_find

    def change_target_and_reset(self, column_to_find: str):
        self.change_target_to(column_to_find)
        self.reset_index()
        self.disable_approx()

    def find_first_index(self, column_to_find: str):

        old_target = self.target
        old_index = self.index

        self.change_target_and_reset(column_to_find)

        return_index = self.__next__()

        self.disable_approx()
        self.set_index_to(old_index)
        self.change_target_to(old_target)

        return return_index


def _filter_file(target_dir):

    if (os.path.isdir(target_dir) is False):

        print(f"Target directory {target_dir} doesn't exist.")
        return 0
    
    # file_list_origin = [".xlsx", "A_BB_CC.xlsx", "A_.xlsx", "XX_A_CC.xlsx", "~$name.xlsx"]

    # os.listdir(target_dir)

    file_list = [
        f"./{target_dir}/{file_name}" for file_name in os.listdir(target_dir)
        if not any(pattern.match(file_name) for pattern in [cache_file_pattern, masked_file_pattern])
        and xlsx_file_pattern.search(file_name)
        and underscore_pattern.search(file_name)
    ]

    print(f"Filtered: {file_list}")

    return file_list


def _preprocess_file(file):

    df_dict = pd.read_excel(file, sheet_name=None)
    name = file.split("/")[-1].split("_")[1]

    result_list = []

    for df in list(df_dict.values()):

        iterator = RowIterator(df.columns)
        iterator.enable_approx()

        result_list.append((iterator, df))

    return (name, result_list)


def preprocess(target_dir):
    
    file_list = _filter_file(target_dir)
    preprocessed_result = []

    for file in file_list:

        preprocessed_result.append(_preprocess_file(file))

    return preprocessed_result


def process_file(iterator_and_df_list, network):

    total_samples = 0
    effective_samples = 0

    for iterator, df in iterator_and_df_list:

        for _, row in df.iterrows():

            effective = process_row(iterator, row, network)

            if effective:
                effective_samples += 1

            total_samples += 1

    return (total_samples, effective_samples)


def process_row(iterator, row, network):
    
    def find_src():
        
        iterator.change_target_and_reset(src_column_name)
        inst_name = None

        for index in iterator:

            cell = row[index]

            if (are_similar_texts(cell, src_target_name) is not None):

                inst_name = normalize_inst_name(row[index + src_offset])
                break

        inst = Inst(inst_name)
        inst.field = network.name

        return inst

    def find_dst():

        iterator.change_target_and_reset(dst_column_name)
        inst_name = None

        for index in iterator:

            cell = row[index]

            if (are_similar_texts(cell, dst_target_name) is not None):

                inst_name = normalize_inst_name(row[index + dst_offset])
                break

        inst = Inst(inst_name)
        inst.field = network.name

        return inst
    
    def find_edge(src, dst):

        if (any((node is None or node.valid() is False) for node in [src, dst])):
            return None
        
        move = Move(src.name, dst.name)

        move.current_rank = row[iterator.find_first_index("Current Rank")]
        move.gender = row[iterator.find_first_index("Sex")]

        return move
         
    src = find_src()
    dst = find_dst()
    edge = find_edge(src, dst)

    all_set = True

    if (src is None or not src.valid()):
        all_set = False
    
    if (dst is None or not dst.valid()):
        all_set = False

    if (edge is None or not edge.valid()):
        all_set = False

    if all_set:

        network.add_inst(src)
        network.add_inst(dst)
        network.add_move(edge)

    return all_set


def construct_network(net_type='global'):

    if net_type not in ['global', 'domestic']:
        return None

    prep_list = preprocess("data")
    network_dict = {}

    for field, df_list in prep_list:
        
        network_dict[field] = Field(name=field)

        print(f"=== Field: {field} ===")

        total, effective = process_file(df_list, network_dict[field])

        print(f"Total samples: {total}")
        print(f"Effective samples: {effective}")

        print(f"Sparsity: {calc_sparsity(network_dict[field])}")
        print(f"Domestic sparsity: {calc_sparsity(network_dict[field].domestic)}")

        print("\n")

    if net_type == 'global':
        return network_dict
    
    for field, net in network_dict.items():
        network_dict[field] = net.domestic

    return network_dict


