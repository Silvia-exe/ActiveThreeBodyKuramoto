import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import plotting as kplt
import numpy as np
import math

def box_muller_noise():
    U1 = np.random.uniform()
    U2 = np.random.uniform()
    R = np.sqrt(-2 * np.log(U1))
    phi = 2 * np.pi * U2
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return (X,Y)

def move_noise(r_v, th):
    new_r_v = np.copy(r_v)
    for i in range(len(r_v)):
        nx, ny = box_muller_noise()
        dx = v*dt*np.cos(th[i]) + 1/gamma * nx
        dy = v*dt*np.sin(th[i]) + 1/gamma * ny
        #dx = v*dt + 1/gamma * nx
        #dy = v*dt + 1/gamma * ny
        new_r_v[i] += [dx, dy]

    return periodic_bc(new_r_v)

#Movement is done by the velocity in the direction of the oscillator angle
def move(r_v, th):
    new_r_v = np.copy(r_v)
    for i in range(len(r_v)):
        new_r_v[i] += [v*dt*np.cos(th[i]), v*dt*np.sin(th[i])]

    return periodic_bc(new_r_v)

#Neighbors are two closest nodes
def kuramoto_int(th, neigh):
    new_th = np.copy(th)
    for i in range(len(th)):
        th_l = th[int(neigh[i,0])]
        th_r = th[int(neigh[i,1])]
        new_th[i] += dt*K1/2*(np.sin(th_l-th[i]) + np.sin(th_r-th[i])) + K2/2 * (np.sin(2*th_r-th_l-th[i])+np.sin(2*th_l-th_r+th[i]))
    return new_th

#N to N connections of oscillators
def kuramoto_global(th):
    new_th = np.copy(th)
    for i in range(len(th)):
        for j in range(len(th)):
            new_th[i] += dt*K1/N*(np.sin(th[j]-th[i]))
    return new_th

#Detects if particle is at boundary and implements periodic boundary conditions
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

#Produces animation only
def animate():
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lx)
    ax.set_aspect('equal', adjustable='box')
    normalize = colors.Normalize(vmin=-1, vmax=1)
    sc = ax.scatter(r_0[:, 0], r_0[:, 1], c=np.sin(th_0), norm = normalize, cmap="viridis", edgecolor="black", s = 100)
    cb = plt.colorbar(sc)

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

    ani = animation.FuncAnimation(fig, update, frames=200, interval=20, blit=True)
    plt.show()

#For animating the produced act_mat
def animate_from_data(filename = None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lx)
    ax.set_aspect('equal', adjustable='box')

    normalize = colors.Normalize(vmin=-1, vmax=1)
    sc = ax.scatter(pos_mat[:, 0, 0], pos_mat[:, 1, 0], c=np.sin(act_mat[:, 0]), norm= normalize, cmap="viridis", edgecolor="black", s = 100)
    cb = plt.colorbar(sc)
    cb.set_label(r'sin($\theta$)')

    def update(frame):
        sc.set_offsets(pos_mat[:, :, frame])
        sc.set_array(np.sin(act_mat[:, frame]))
        return sc,

    ani = animation.FuncAnimation(fig, update, frames=act_mat.shape[1], interval=20, blit=True)

    if filename is not None:
        ani.save(filename = "../"+filename + "anim.gif", writer = "pillow")
    #plt.show()

def plot_from_data(filename = None):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 5),
                         subplot_kw={
                             "ylim": (0, Lx),
                             "xlim": (0, Lx),
                             "aspect": "equal",
                             "adjustable": "box"})

    times = [0, int(act_mat.shape[1]/2), int(act_mat.shape[1])-1]
    normalize = colors.Normalize(vmin=-1, vmax=1)

    for ax, time in zip(ax, times):
        sc = ax.scatter(pos_mat[:, 0, time],
                pos_mat[:, 1, time],
                c=np.sin(act_mat[:, time]), cmap="viridis", norm = normalize, edgecolor="black", s = 100)
        ax.set_title(f'Time = {time}')


    fig.subplots_adjust(right=0.8)
    cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

    cb = fig.colorbar(sc, cax=cb_ax)
    cb.set_label(r'sin($\theta$)')

    plt.suptitle(f"Time evol. for system with $K_1$= {K1} , $K_2$ = {K2} and $v_0$ = {v}" )

    if filename is not None:
        plt.savefig("../"+filename+"_seq.png")

    #plt.show()


N= 100#number of particles
T = 150
Lx = 10 #Box size
K1_v = [1,0,1,3,1] #Pairwise interaction parameter
K2_v = [0,1,1,1,3] #Trio-wise interaction parameter
dt = 0.03 #Integration step
v = 1 #Internal particle velocity
gamma = 10
t = np.linspace(0,T,int(T/dt))

#kplt.animate_oscillators(act_mat)

for j in range(len(K1_v)):
    K1 = K1_v[j]
    K2 = K2_v[j]

    r_0, th_0 = initialize_r0()
    neigh = find_neighbors(r_0)
    act_mat = np.reshape(th_0, (N,1))
    pos_mat = np.zeros((N, 2, len(t)+1))
    pos_mat[:, :, 0] = r_0

    for i,_ in enumerate(t):
        th_0 = kuramoto_int(th_0, neigh)
        r_0 = move_noise(r_0,th_0)
        neigh = find_neighbors(r_0)
        pos_mat[:, :, i+1] = r_0
        act_mat = np.column_stack((act_mat,th_0))

    filename = "K1_"+str(K1)+"_K2_"+str(K2)+"_v_"+str(v)
    kplt.plot_phase_coherence(act_mat,filename)
    animate_from_data(filename)
    plot_from_data(filename)

#kplt.plot_phase_coherence(act_mat)
#animate_from_data()
#plot_from_data()

#animate()
