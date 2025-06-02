import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotting as kplt
import numpy as np
import math

N= 100#number of particles
T = 50
Lx = 10 #Box size
K1 = 1.0 #Pairwise interaction parameter
K2 = 0.0 #Trio-wise interaction parameter
dt = 0.05 #Integration step
v = 1.0 #Internal particle velocity
t = np.linspace(0,T,int(T/dt))

def move(r_v, th):
    new_r_v = np.copy(r_v)
    for i in range(len(r_v)):
        new_r_v[i] += [v*dt*np.cos(th[i]), v*dt*np.sin(th[i])]

    return periodic_bc(new_r_v)


def kuramoto_int(th, neigh):
    new_th = np.copy(th)
    for i in range(len(th)):
        th_l = th[int(neigh[i,0])]
        th_r = th[int(neigh[i,1])]
        new_th[i] += dt*K1/2*(np.sin(th_l-th[i]) + np.sin(th_r-th[i])) + K2/2 * (np.sin(2*th_r-th_l-th[i])+np.sin(2*th_l-th_r+th[i]))
    return new_th

def kuramoto_global(th):
    new_th = np.copy(th)
    for i in range(len(th)):
        for j in range(len(th)):
            new_th[i] += dt*K1/N*(np.sin(th[j]-th[i]))
    return new_th

#Detects if particle| is at boundary and implements periodic boundary conditions
def periodic_bc(r_new):
    #Detect boundary crossing and implement PBC
    for p in range(N):
        for i in range(2):
            if r_new[p,i] < 0:
                r_new[p,i] += Lx
            elif r_new[p,i] > Lx:
                r_new[p,i] -= Lx

    return r_new

#Initializes N particles with N oscillators
def initialize_r0():
    r0_v = np.random.rand(N,2)*Lx #Create two random numbers between [0,1)
    th_0 = np.random.uniform(-np.pi, np.pi, N)

    return r0_v, th_0

#Find two closest particles of each particle
def find_neighbors(r_v):
    r_vt = np.copy(r_v)
    neigh = np.zeros((len(r_v),2))
    temp = np.zeros(len(r_v))
    for i in range(len(r_v)):
        r_i = r_vt[i]
        for j in range(len(r_v)):
            if j != i:
                temp[j] = math.hypot(r_i[0]-r_vt[j,0], r_i[1]-r_vt[j,1])
            else: temp[j] = Lx

        neigh[i,0] = np.argmin(temp)
        temp[np.argmin(temp)] = 2*Lx #Biggest system number
        neigh[i,1] = np.argmin(temp)
    return neigh

def animate():
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lx)
    ax.set_aspect('equal', adjustable='box')
    sc = ax.scatter(r_0[:, 0], r_0[:, 1], c=np.sin(th_0), cmap="viridis", edgecolor="black")
    cb = plt.colorbar(sc)
    # --- Animation update function ---
    def update(frame):
        global th_0, r_0, neigh
        th_0 = kuramoto_int(th_0,neigh)
        r_0 = move(r_0,th_0)
        neigh = find_neighbors(r_0)
        #print(th_0)
        ang_c = np.sin(th_0)

        sc.set_offsets(r_0)
        sc.set_array(ang_c)
        return sc,

    # --- Run animation ---
    ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)
    plt.show()

def animate_from_data():
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lx)
    ax.set_aspect('equal', adjustable='box')

    sc = ax.scatter(pos_mat[:, 0, 0], pos_mat[:, 1, 0],
                    c=np.sin(act_mat[:, 0]), cmap="viridis", edgecolor="black")
    plt.colorbar(sc)

    def update(frame):
        #sc.set_offsets(pos_mat[:, :, frame])
        sc.set_array(np.sin(act_mat[:, frame]))
        return sc,

    ani = animation.FuncAnimation(fig, update, frames=act_mat.shape[1], interval=50, blit=True)
    plt.show()

r_0, th_0 = initialize_r0()
neigh = find_neighbors(r_0)
act_mat = np.reshape(th_0, (N,1))
pos_mat = np.zeros((N, 2, len(t)+1))
pos_mat[:, :, 0] = r_0

for i,_ in enumerate(t):
    th_0 = kuramoto_global(th_0)#, neigh)
    r_0 = move(r_0,th_0)
    neigh = find_neighbors(r_0)
    pos_mat[:, :, i+1] = r_0
    act_mat = np.column_stack((act_mat,th_0))

kplt.plot_phase_coherence(act_mat)
kplt.animate_oscillators(act_mat)

animate_from_data()


#animate()
