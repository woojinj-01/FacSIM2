from facsimlib.processing import construct_network

net_global = construct_network()
net_domestic = construct_network(net_type='domestic')

for net in net_global.items():
    print(net)

for net in net_domestic.items():
    print(net)

