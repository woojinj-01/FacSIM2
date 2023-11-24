import networkx as nx
import facsimlib.SpringRank as sp
from facsimlib.text import get_country_code, get_region, normalize_inst_name

from copy import deepcopy
from random import shuffle
import numpy as np
import pandas as pd


class Field():
    def __init__(self, name, based_on_graph=None, nx_data=None, **attr) -> None:
        
        if (based_on_graph is not None):
            self.net = based_on_graph
            self.net.name = name
        else:
            self.net = nx.MultiDiGraph(name=name, incoming_graph_data=nx_data, **attr)

        self.inst_id = 1

    def __repr__(self):

        str_list = []

        str_list.append(f"Name: {self.name}\n")

        str_list.append(f"Institutions: {len(list(self.net.nodes))}\n")
        
        for inst_name in list(self.net.nodes):
            str_list.append(inst_name)

        str_list.append("\n")

        str_list.append(f"Moves: {len(list(self.net.edges))}\n")
        
        for move in list(self.net.edges):
            str_list.append(f"{move[0]} -> {move[1]}")

        str_list.append("\n")

        return "\n".join(str_list)

    @property
    def name(self):
        return self.net.name
    
    @name.setter
    def name(self, name):
        self.net.name = name

    @property
    def closed(self):

        self_closed = self.filter("country_code", "KR")
        self_closed.name = f"Closed {self.name}"

        return self_closed
    
    @property
    def random(self):
        self_random = self.randomize()[0]
        self_random.name = f"Random {self.name}"

        return self_random
    
    @property
    def collected(self):    # to be implemented
        pass

    @property
    def rank_length(self):

        length = 0

        for name in list(self.net.nodes):

            inst = self.inst(name)

            if inst:
                if (inst['rank']):
                    length += 1

        return length
    
    def ranks(self, inverse=False, normalized=False):

        ranks = {}

        for name in list(self.net.nodes):

            inst = self.inst(name)

            if (inst is None):
                continue
                
            if (inverse is True):
                ranks[name] = inst['rank']

            else:
                ranks[inst['rank']] = name

        if normalized is True:
            if inverse is True:
                return {key: value / len(ranks) for key, value in ranks.items()}
            else:
                return {key / len(ranks): value for key, value in ranks.items()}
        else:
            return ranks    
        
    def inst(self, name):

        name_norm = normalize_inst_name(name)

        if (name_norm is None):
            return None

        if (self.net.has_node(name_norm)):
            return self.net._node[name_norm]
        
        else:
            return None
        
    def move(self, u_name, v_name):
        
        u_name_norm = normalize_inst_name(u_name)
        v_name_norm = normalize_inst_name(v_name)

        if (any(name is None for name in [u_name_norm, v_name_norm])):
            return None
        
        if (self.net.has_edge(u_name_norm, v_name_norm)):
            return self.net[u_name_norm][v_name_norm]

    def add_inst(self, inst):

        if (self.net.has_node(inst.name)):
            return None

        self.net.add_node(inst.name, id=self.inst_id, rank=None, **inst.to_dict())

        self.inst_id += 1

        return self.net.nodes[inst.name]

    def add_move(self, move):

        self.net.add_edge(move.u_name, move.v_name, **move.to_dict())

        return 1
    
    def set_ranks(self):

        self.reset_ranks()

        adj_mat = nx.to_numpy_array(self.net)

        adj_mat += 1

        scores = sp.get_ranks(adj_mat)

        sorted_indices = np.argsort(scores)[::-1]

        value_to_rank = {value: rank + 1 for rank, value in enumerate(sorted_indices)}

        integers_based_on_rank = [value_to_rank[value] for value in range(len(scores))]

        integers_based_on_rank.reverse()

        for node in self.net.nodes():

            self.net.nodes[node]['rank'] = integers_based_on_rank.pop()

        return self

    def reset_ranks(self):

        for node in self.net.nodes():

            self.net.nodes[node]['rank'] = None

        return self

    def copy_ranks_from(self, another_field):

        ranks_to_copy = {}

        for node in another_field.net.nodes():

            if (another_field.net.nodes[node]['rank'] is not None):

                ranks_to_copy[node] = another_field.net.nodes[node]['rank']

        for node in list(ranks_to_copy.keys()):

            self.net.nodes[node]['rank'] = ranks_to_copy[node]

        return self

    def filter(self, key, value, op='='):

        op_allowed = ['=', 'in', '!in']
        
        if op not in op_allowed:
            return None
        
        match op:
            case '=':
                return self._filter_equals(key, value)

            case 'in':
                return self._filter_containing(key, value)
            
            case '!in':
                return self._filter_not_containing(key, value)
    
    def _filter_equals(self, key, value):
        
        net_filtered = deepcopy(self.net)

        nodes_to_remove = []

        for node in net_filtered.nodes():
            if key in net_filtered.nodes[node] and net_filtered.nodes[node][key] != value:

                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            net_filtered.remove_node(node)

        return Field(f"Filtered {self.name} ({key} = {value})", net_filtered)

    def _filter_containing(self, key, value):
        
        net_filtered = deepcopy(self.net)

        nodes_to_remove = []

        for node in net_filtered.nodes():
            if key in net_filtered.nodes[node] and net_filtered.nodes[node][key] not in value:

                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            net_filtered.remove_node(node)

        return Field(f"Filtered {self.name} ({key} in {value})", net_filtered)
    
    def _filter_not_containing(self, key, value):
        
        net_filtered = deepcopy(self.net)

        nodes_to_remove = []

        for node in net_filtered.nodes():
            if key in net_filtered.nodes[node] and net_filtered.nodes[node][key] in value:

                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            net_filtered.remove_node(node)

        return Field(f"Filtered {self.name} ({key} not in {value})", net_filtered)
    
    def randomize(self, times=1):

        random_nets = []

        for i in range(times):

            net = self._randomize()
            net.name = f"Random {self.name} {i}"

            random_nets.append(net)

        return random_nets
    
    def _randomize(self):

        net_rand = deepcopy(self.net)

        inst_names_u = []
        inst_names_v = []

        for name, deg in self.net.out_degree:
            
            for _ in range(deg):
                inst_names_u.append(name)

        for name, deg in self.net.in_degree:

            for _ in range(deg):
                inst_names_v.append(name)

        net_rand.clear_edges()

        shuffle(inst_names_u)
        shuffle(inst_names_v)

        for inst_u, inst_v in zip(inst_names_u, inst_names_v):
            
            net_rand.add_edge(inst_u, inst_v)

        return Field(f"Random {self.name}", net_rand)
    
    def export_ranks(self):

        path = f"ranks_{self.name}.csv"

        data = {"Rank": [], "Inst": []}

        for rank in sorted(self.ranks().keys()):

            data["Rank"].append(rank)
            data["Inst"].append(self.ranks()[rank])

        df = pd.DataFrame(data)
        
        return df.to_csv(path, sep='\t', index=False)
    

class Institution():
    def __init__(self, inst_name: str) -> None:

        self.name = inst_name
        self.field = None

        self.region = get_region(inst_name)
        self.country_code = get_country_code(inst_name)

    def __repr__(self) -> str:
        
        str_list = []

        str_list.append(f"Name: {self.name}")

        for name, value in vars(self).items():
            str_list.append(f"{name}: {value}")

        return "\n".join(str_list)
    
    def valid(self):

        if (all(value is not None for value in vars(self).values())):
            return True
        else:
            return False
        
    def to_dict(self):

        assert (self.valid())

        result = {}

        for name, value in vars(self).items():
            result[name] = value

        return result
    
    
class Move():
    def __init__(self, inst_name_u: str, inst_name_v: str) -> None:
        
        self.u_name = inst_name_u
        self.v_name = inst_name_v

        self.current_rank = None
        self.gender = None

    def __repr__(self) -> str:
        
        str_list = []

        for name, value in vars(self).items():
            str_list.append(f"{name}: {value}")

        return "\n".join(str_list)

    def valid(self):

        if all(value is not None for value in vars(self).values()):
            return True
        
        else:
            return False
        
    def to_dict(self):

        assert (self.valid())

        result = {}

        for name, value in vars(self).items():
            result[name] = value

        return result