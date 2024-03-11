import networkx as nx
import facsimlib.SpringRank as sp
from facsimlib.text import get_country_code, get_region, normalize_inst_name
import facsimlib.processing

from copy import deepcopy
from random import shuffle
import numpy as np
import pandas as pd


class NodeSelect:
    def __init__(self, key, value, op, label=None) -> None:
        
        op_allowed = ['=', 'in', '!in']

        if op not in op_allowed:
            return None

        self.key = key
        self.value = value
        self.op = op

        self.label = label

    def __repr__(self) -> str:
        
        match self.op:

            case '=':
                return f"{self.key}={self.value}"

            case 'in':
                return f"{self.key} in {self.value}"

            case '!in':
                return f"{self.key} not in {self.value}"
            
            case _:
                return "Invalid"
    
    def hit(self, node):

        if self.key not in node:
            return False

        target = node[self.key]

        match self.op:

            case '=':
                if target == self.value:
                    return True

            case 'in':
                if target in self.value:
                    return True

            case '!in':
                if target not in self.value:
                    return True

        return False
        

class EdgeSelect:
    def __init__(self, key, value, op) -> None:
        pass


class Field():
    def __init__(self, name, based_on_graph=None, nx_data=None, color=None, net_type='global', **attr) -> None:
        
        if (based_on_graph is not None):
            self.net = based_on_graph
            self.net.name = name
        else:
            self.net = nx.MultiDiGraph(name=name, incoming_graph_data=nx_data, **attr)

        self.inst_id = 1

        self.stats = {"region": set(),
                      "country": set()}
        
        if net_type == 'global':
            self.type = 'global'
        
        elif net_type == 'domestic':
            self.type = 'domestic'

        else:
            return None

        match self.name:

            case "Biology":
                self.color = "#0584F2"

            case "Computer Science":
                self.color = "#F18904"

            case "Physics":
                self.color = "#00743F"

            case _:

                if color is None:
                    self.color = "black"

                else:
                    self.color = color

    def __repr__(self):

        str_list = []

        str_list.append(f"Name: {self.name}\n")

        str_list.append(f"Institutions: {len(list(self.net.nodes))}\n")
        
        # for inst_name in list(self.net.nodes):
        #     str_list.append(inst_name)

        str_list.append("\n")

        str_list.append(f"Moves: {len(list(self.net.edges))}\n")
        
        # for move in list(self.net.edges):
        #     str_list.append(f"{move[0]} -> {move[1]}")

        str_list.append("\n")

        return "\n".join(str_list)

    @property
    def name(self):
        return self.net.name
    
    @name.setter
    def name(self, name):
        self.net.name = name

    @property
    def domestic(self):

        select = NodeSelect("country_code", "KR", "=")

        self_domestic = self.filter(select)
        self_domestic.name = f"Domestic {self.name}"
        self_domestic.net_type = 'domestic'

        return self_domestic
    
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
                if (inst['rank'] and inst['rank_norm']):
                    length += 1

        return length
    
    def ranks(self, inverse=False, normalized=False):

        ranks = {}

        for name in list(self.net.nodes):

            inst = self.inst(name)

            rank = inst['rank_norm'] if normalized else inst['rank']

            if (inst is None):
                continue
                
            if (inverse is True):
                ranks[name] = rank

            else:
                ranks[rank] = name

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

        self.stats["region"].add(get_region(inst.name))
        self.stats["country"].add(get_country_code(inst.name))

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
            self.net.nodes[node]['rank_norm'] = self.net.nodes[node]['rank'] / len(self.net.nodes())

        return self

    def reset_ranks(self):

        for node in self.net.nodes():

            self.net.nodes[node]['rank'] = None
            self.net.nodes[node]['rank_norm'] = None

        return self

    def copy_ranks_from(self, another_field):

        ranks_to_copy = {}
        ranks_norm_to_copy = {}

        for node in another_field.net.nodes():

            if (another_field.net.nodes[node]['rank'] is not None and another_field.net.nodes[node]['rank_norm'] is not None):

                ranks_to_copy[node] = another_field.net.nodes[node]['rank']
                ranks_norm_to_copy[node] = another_field.net.nodes[node]['rank_norm']

        for node in list(ranks_to_copy.keys()):

            self.net.nodes[node]['rank'] = ranks_to_copy[node]
            self.net.nodes[node]['rank_norm'] = ranks_norm_to_copy[node]

        return self

    def filter(self, select: NodeSelect):

        net_filtered = deepcopy(self.net)

        nodes_to_remove = []

        for node in net_filtered.nodes():
            if not select.hit(net_filtered.nodes[node]):

                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            net_filtered.remove_node(node)

        return Field(f"Filtered {self.name} ({select})", net_filtered, color=self.color)
    
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

        return Field(f"Random {self.name}", net_rand, color=self.color)
    
    def export_ranks(self):

        path = f"ranks_{self.name}.csv"

        data = {"Rank": [], "Inst": []}

        for rank in sorted(self.ranks().keys()):

            data["Rank"].append(rank)
            data["Inst"].append(self.ranks()[rank])

        df = pd.DataFrame(data)
        
        return df.to_csv(path, sep='\t', index=False)
    
    def export_stats(self, key):

        if key not in self.stats.keys():
            return

        data = self.stats[key]
        
        path = f"stats_{key}_{self.name}.csv"

        df = pd.DataFrame(data)
        
        return df.to_csv(path, sep='\t', index=False)
    
    def export_node_list(self):

        path = f"stats_node_{self.name}.csv"
        
        nodes = list(self.net.nodes())

        df = pd.DataFrame(nodes)

        return df.to_csv(path, sep='\t', index=False)

    def export_edge_list(self):

        path = f"stats_edge_{self.name}.csv"
        
        edges = list(self.net.edges())

        df = pd.DataFrame(edges)

        return df.to_csv(path, sep='\t', index=False)

    

class Institution:
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
    
    
class Move:
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
    

if __name__ == "__main__":

    network_dict = facsimlib.processing.construct_network()

    for net in network_dict.values():

        net_c = net.domestic.set_ranks()
        net.set_ranks()

        net.export_ranks()
        net_c.export_ranks()