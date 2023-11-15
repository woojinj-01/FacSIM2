from facsimlib.academia import Institution as Inst
from facsimlib.academia import Move
import facsimlib.academia


if (__name__ == "__main__"):

    node_info = {"Type": "University", "Rank": 3}
    edge_info = {"Type": "Ph.D."}

    field = "CS"

    node1 = Inst("KAIST")
    node1.field = field

    node2 = Inst("SNU")
    node2.field = field

    node3 = Inst("KU")
    node3.field = field

    edge1 = Move("KAIST", "SNU")
    edge1.current_rank = "Assitant Professor"
    edge1.gender = "Male"

    edge2 = Move("KAIST", "SNU")
    edge2.current_rank = "Assitant Professor"
    edge2.gender = "Female"

    edge3 = Move("SNU", "KU")
    edge3.current_rank = "Assitant Professor"
    edge3.gender = "Feale"

    net = facsimlib.academia.Field(name="Test 1")

    net.add_inst(node1)
    net.add_inst(node2)
    net.add_inst(node3)

    net.add_move(edge1)
    net.add_move(edge2)
    net.add_move(edge3)

    # net_copied = copy.deepcopy(net)

    print(net)

    print("=" * 20)

    net_copied = net.randomize()
    print(net_copied)