
from kuramoto import Kuramoto
import plotting as kplt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#Generates a 2D array of 1st degree neighbors of a 1D circle of oscillators inforcing PBC
def adjacency_matrix1D(N):
    adj_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j == i-1 or j == i+1:
                adj_mat[i][j] = 1 #Immediate 1st degree neighbors

    #Forcing PBC, a circle
    adj_mat[0,-1] = adj_mat[-1,0] = 1

    return adj_mat

def main():
    N = 100#Number of oscillators
    T = 100
    dt = 0.01
    coupling = 5
    #nat_freq = np.zeros(N)
    adj_mat = adjacency_matrix1D(N)

    #adj_mat = np.ones((N,N))
    #np.fill_diagonal(adj_mat,0)
    #print(adj_mat)

    # Instantiate model with parameters
    model = Kuramoto(coupling, dt, T, n_nodes=len(adj_mat))

    # Run simulation - output is time series for all nodes (node vs time)
    act_mat = model.run1D_trio(adj_mat=adj_mat)
    #print(act_mat.shape)
    #print(len(act_mat[0]))
    # Plot all the time series
    kplt.plot_activity(act_mat)
    kplt.plot_phase_coherence(act_mat)
    #kplt.oscillators_comp(act_mat)
    kplt.animate_oscillators(act_mat)

if __name__ == '__main__':
    main()
