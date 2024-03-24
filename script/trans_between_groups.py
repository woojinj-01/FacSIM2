import numpy as np

from facsimlib.processing import construct_network
from facsimlib.math import trans_prob_between_regions


if __name__ == "__main__":

    nets = construct_network()

    trial = 10
    
    for net in nets.values():
        
        trans_mat = trans_prob_between_regions(net)

        print(net.name)
        print(trans_mat)

        randomized = net.randomize(trial)
        mat_randomized = []

        for net_r in randomized:

            mat_randomized.append(trans_prob_between_regions(net_r))

        random_avg = np.mean(mat_randomized, axis=0)

        print(f"Ranomized {net.name}")
        print(random_avg)

        print("Difference")
        print(trans_mat - random_avg)