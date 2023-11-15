from facsimlib.academia import Institution as Inst
from facsimlib.academia import Move
import facsimlib.academia


if (__name__ == "__main__"):

    node_info = {"Type": "University", "Rank": 3}
    edge_info = {"Type": "Ph.D."}

    node1 = Inst("KAIST", {"Type": "University", "Rank": 3})
    node2 = Inst("SNU", {"Type": "University", "Rank": 1})

    edge = Move("KAIST", "SNU", edge_info)

    net = facsimlib.academia.Field(name="Test 1")

    net.add_inst(node1)
    net.add_inst(node2)

    net.add_move(edge)

    print(net.__repr__())
