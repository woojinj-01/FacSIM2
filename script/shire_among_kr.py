from facsimlib.processing import construct_network
from facsimlib.academia import NodeSelect as NS
from facsimlib.text import con_america, con_europe, con_ocenaia, con_asia_without_kr
from facsimlib.plot import param_alpha, split_color_by

import numpy as np
import matplotlib.pyplot as plt


def get_shire_among_kr_doctorates(network):

    num_group = 10

    total_count = {}

    kr_count = {}
    asia_count = {}
    america_count = {}
    europe_count = {}
    oceania_count = {}

    shire_kr_count = {}

    ns_korea = NS('country_code', ["KR", "KOREA"], 'in', label="Korea")
    ns_asia = NS('country_code', con_asia_without_kr, 'in', label="Asia")
    ns_america = NS('country_code', con_america, 'in', label="America")
    ns_europe = NS('country_code', con_europe, 'in', label="Europe")
    ns_oceania = NS('country_code', con_ocenaia, 'in', label="Europe")

    for node in network.net.nodes:

        edges_in = network.net.in_edges(node, data=True)

        rank = network.inst(node)['rank']

        if (rank is None):
            continue

        total_count[rank] = len(edges_in)
        kr_count[rank] = 0
        asia_count[rank] = 0
        america_count[rank] = 0
        europe_count[rank] = 0
        oceania_count[rank] = 0

        shire_kr_count[rank] = 0

        for src, dst, data in edges_in:

            if (ns_korea.hit(network.net.nodes[src])):
                kr_count[rank] += 1

                if src == dst:
                    shire_kr_count[rank] += 1

            elif ((ns_asia.hit(network.net.nodes[src]))):
                asia_count[rank] += 1

            elif ((ns_america.hit(network.net.nodes[src]))):
                america_count[rank] += 1

            elif ((ns_europe.hit(network.net.nodes[src]))):
                europe_count[rank] += 1

            elif ((ns_oceania.hit(network.net.nodes[src]))):
                oceania_count[rank] += 1

    ranks_sorted = [key for key, _ in sorted(total_count.items())]
    kr_count_sorted = [value for _, value in sorted(kr_count.items())]
    asia_count_sorted = [value for _, value in sorted(asia_count.items())]
    america_count_sorted = [value for _, value in sorted(america_count.items())]
    europe_count_sorted = [value for _, value in sorted(europe_count.items())]
    oceania_count_sorted = [value for _, value in sorted(oceania_count.items())]

    shire_kr_count_sorted = [value for _, value in sorted(shire_kr_count.items())]

    grousize = np.floor(len(ranks_sorted) / num_group)

    if (grousize == 1):

        x_co = ranks_sorted
        y_co_kr = kr_count_sorted
        y_co_asia = asia_count_sorted
        y_co_america = america_count_sorted
        y_co_europe = europe_count_sorted
        y_co_oceania = oceania_count_sorted
        y_co_shire_kr = shire_kr_count_sorted

    else:
        
        x_co = []

        y_co_kr = []
        y_co_asia = []
        y_co_america = []
        y_co_europe = []
        y_co_oceania = []
        y_co_shire_kr = []
        
        index = 0
        grouid = 1

        while index < num_group * grousize:

            elements_kr = []
            elements_asia = []
            elements_america = []
            elements_europe = []
            elements_oceania = []
            
            elements_shire_kr = []

            while (grousize > len(elements_kr)):

                elements_kr.append(kr_count_sorted[index])
                elements_asia.append(asia_count_sorted[index])
                elements_america.append(america_count_sorted[index])
                elements_europe.append(europe_count_sorted[index])
                elements_oceania.append(oceania_count_sorted[index])

                elements_shire_kr.append(oceania_count_sorted[index])

                index += 1

                if (index >= len(ranks_sorted)):
                    break

            x_co.append(grouid)

            y_co_kr.append(sum(elements_kr))
            y_co_asia.append(sum(elements_asia))
            y_co_america.append(sum(elements_america))
            y_co_europe.append(sum(elements_europe))
            y_co_oceania.append(sum(elements_oceania))

            y_co_shire_kr.append(sum(elements_shire_kr))

            grouid += 1

        while index < len(ranks_sorted):

            y_co_kr[-1] += kr_count_sorted[index]
            y_co_asia[-1] += asia_count_sorted[index]
            y_co_america[-1] += america_count_sorted[index]
            y_co_europe[-1] += europe_count_sorted[index]
            y_co_oceania[-1] += oceania_count_sorted[index]
            
            y_co_shire_kr[-1] += shire_kr_count_sorted[index]

            index += 1

    y_co_kr_norm = [y_co_kr[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                    if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]
    
    y_co_asia_norm = [y_co_asia[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                      if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]
    
    y_co_america_norm = [y_co_america[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                         if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]
    
    y_co_europe_norm = [y_co_europe[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                        if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]
    
    y_co_oceania_norm = [y_co_oceania[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                         if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]

    y_co_shire_kr_norm = [y_co_shire_kr[i] / (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i])\
                          if (y_co_kr[i] + y_co_asia[i] + y_co_america[i] + y_co_europe[i] + y_co_oceania[i]) != 0 else 0 for i in range(len(x_co))]

    y_co_kr = y_co_kr_norm
    y_co_asia = y_co_asia_norm
    y_co_america = y_co_america_norm
    y_co_europe = y_co_europe_norm
    y_co_oceania = y_co_oceania_norm

    y_co_shire_kr = y_co_shire_kr_norm

    for i in range(len(y_co_kr)):

        print_out_ratio(i + 1, y_co_kr[i], y_co_asia[i], y_co_america[i], y_co_europe[i], y_co_oceania[i], y_co_shire_kr[i])
        plot_ratio(y_co_kr, y_co_shire_kr, network.color, network.name)


def print_out_ratio(group, kr, asia, america, europe, oceania, shire_kr):
    
    print(f"Group {group}")

    print("Ratio of...")

    print(f"S. Korea: {kr}")
    print(f"Asia: {asia}")
    print(f"America: {america}")
    print(f"Europe: {europe}")
    print(f"Oceania: {oceania}")

    print(f"Self-hires / Total hires: {shire_kr}")

    if kr != 0:
        print(f"Self-hires / S.Korea hires: {shire_kr / kr}")


def plot_ratio(kr, shire_kr, color, name):

    if len(kr) != len(shire_kr):
        return None
    
    fig_name = f"./fig/kr_shire_{name}.pdf"
    
    palette = split_color_by(color, 2)
    
    num_groups = len(kr)

    x_co = [f"Group\n{i + 1}" for i in range(num_groups)]

    shire_among_kr = [shire_kr[i] / kr[i] if kr[i] != 0 else 0 for i in range(num_groups)]

    kr_minus_shire = [kr[i] - shire_kr[i] for i in range(num_groups)]

    plt.bar(x_co, shire_kr, label="Self hires / Total hires", color=palette[0])
    plt.bar(x_co, kr_minus_shire, label="KR hires / Total hires", color=palette[1], bottom=shire_kr)
    
    plt.plot(x_co, shire_among_kr, 'rs--')

    plt.legend()

    plt.savefig(fig_name, bbox_inches='tight')

    plt.clf()


if __name__ == "__main__":

    nets = construct_network()
    
    for net in nets.values():
        net.copy_ranks_from(net.domestic.set_ranks())

        print(net.name)

        get_shire_among_kr_doctorates(net)