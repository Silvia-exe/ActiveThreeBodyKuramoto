
from kuramoto import Kuramoto
import plotting as kplt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#Generates a 2D array of 1st degree neighbors of a 1D circle of oscillators inforcing PBC
def adjacency_1neigh_1D(N):
    adj_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j == i-1 or j == i+1:
                adj_mat[i][j] = 1 #Immediate 1st degree neighbors

    #Forcing PBC, a circle
    adj_mat[0,-1] = adj_mat[-1,0] = 1

    return adj_mat

#Generates a 2D array of all-to-all connectivity of oscillators.
def adjacency_NtoN_1D(N):
    adj_mat = np.ones((N,N))
    np.fill_diagonal(adj_mat,0)

    return adj_mat

def main():
    N = 3 #Number of oscillators
    T = 10 #Simulation time
    dt = 0.01 #Integration time step
    coupling_vals = np.linspace(0, 10, 5) #Value(s) of coupling weights.
    runs = [] #Angle data for the run for each coupling value
    nat_freq = np.zeros(N) #Natural frequency of the oscillators
    adj_mat = adjacency_1neigh_1D(N) #Adjacency matrix


    for coupling in coupling_vals:
        filename = "1D_trio_modif_" + str(N)+ "_oscillators_" + str(coupling) + "_coupling"
        model = Kuramoto(coupling=coupling, dt=dt, T=T, n_nodes=N)
        model.natfreqs = np.random.normal(size=N)  # reset natural frequencies
        act_mat = model.run1D_trio(adj_mat=adj_mat)
        runs.append(act_mat)

        kplt.plot_activity(act_mat, filename)
        kplt.plot_phase_coherence(act_mat, filename)
        kplt.animate_oscillators(act_mat,filename)

if __name__ == '__main__':
    main()
