
from kuramoto import Kuramoto
import plotting as kplt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#Generates a 2D array of 1st degree neighbors of a 1D circle of oscillators inforcing PBC
def adjacency_1neigh_1D(N):
    adj_mat = np.zeros((N,N))
    for i in range(N):
        adj_mat[i][(i+1)%N] = 1 #Immediate 1st degree neighbors
        adj_mat[i][(i-1)%N] = 1 #Immediate 1st degree neighbors

    return adj_mat

def adjacency_triangle_1D(N):

    adj_mat= np.zeros((N,N))
    for i in range(N):
        adj_mat[i][(i+1)%N] = 1 #Immediate 1st degree neighbors
        adj_mat[i][(i-1)%N] = 1 #Immediate 1st degree neighbors
        adj_mat[i][(i+2)%N] = 1 #Immediate 2nd degree neighbors
        adj_mat[i][(i-2)%N] = 1 #Immediate 2nd degree neighbors

    adj_mat_triangle = np.array([adj_mat for _ in range(N)])

    for i in range(N):
        for j in range(N):
            for l in range(N):
                adj_mat_triangle[i][j][l] = adj_mat[i][j]*adj_mat[j][l]*adj_mat[l][i] #Immediate 1st degree neighbors

    return adj_mat_triangle

#Generates a 2D array of all-to-all connectivity of oscillators.
def adjacency_NtoN_1D(N):
    adj_mat = np.ones((N,N))
    np.fill_diagonal(adj_mat,0)

    return adj_mat

def main():
    N = 6 #Number of oscillators
    T = 10 #Simulation time
    dt = 0.01 #Integration time step
    coupling = 2.0 #Value(s) of coupling weights.
    runs = [] #Angle data for the run for each coupling value
    alpha = 0 #For alpha = 0, no threewise interactions. For alpha = 1, only threewise interactions. Use with run_trio_pair
    #nat_freq = np.zeros(N) #Natural frequency of the oscillators
    adj_mat = adjacency_1neigh_1D(N) #Adjacency matrix
    adj_mat_triangle= adjacency_triangle_1D(N)

    model = Kuramoto(coupling=coupling, dt=dt, T=T, n_nodes=N)
    model.natfreqs = np.zeros(N)  # reset natural frequencies
    act_mat = model.run_trio_pair(adj_mat=adj_mat, adj_mat_triangle= adj_mat_triangle)

    kplt.plot_activity(act_mat)
    kplt.plot_phase_coherence(act_mat)
    kplt.animate_oscillators(act_mat)


    '''for coupling in coupling_vals:
        filename = "1D_trio_modif_" + str(N)+ "_oscillators_" + str(coupling) + "_coupling"
        model = Kuramoto(coupling=coupling, dt=dt, T=T, n_nodes=N)
        model.natfreqs = np.random.normal(size=N)  # reset natural frequencies
        act_mat = model.run1D_trio(adj_mat=adj_mat)
        runs.append(act_mat)

        kplt.plot_activity(act_mat, filename)
        kplt.plot_phase_coherence(act_mat, filename)
        kplt.animate_oscillators(act_mat,filename)'''

if __name__ == '__main__':
    main()
