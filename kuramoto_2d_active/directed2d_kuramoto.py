import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
import plotting as kplt
import numpy as np
import math

import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe'

def box_muller_noise(): #Done
    U1 = np.random.uniform()
    U2 = np.random.uniform()
    R = np.sqrt(-2 * np.log(U1))
    phi = 2 * np.pi * U2
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return (X,Y)

def move_noise(r_v, th): #Done
    new_r_v = np.copy(r_v)
    for i in range(len(r_v)):
        nx, ny = box_muller_noise()
        dx = v*dt*np.cos(th[i]) + (2*1/gamma*dt) * nx
        dy = v*dt*np.sin(th[i]) + (2*1/gamma*dt) * ny
        #dx = v*dt + 1/gamma * nx
        #dy = v*dt + 1/gamma * ny
        new_r_v[i] += [dx, dy]

    return periodic_bc(new_r_v)

#Movement is done by the velocity in the direction of the oscillator angle
def move(r_v, th): #Done
    new_r_v = np.copy(r_v)
    for i in range(len(r_v)):
        new_r_v[i] += [v*dt*np.cos(th[i]), v*dt*np.sin(th[i])]

    return periodic_bc(new_r_v)

#Neighbors are two closest nodes
def kuramoto_neigh(th, neigh): #Done
    new_th = np.copy(th)
    for i in range(len(th)):
        th_l = th[int(neigh[i,0])]
        th_r = th[int(neigh[i,1])]
        new_th[i] += dt*(K1/2*(np.sin(th_l-th[i]) + np.sin(th_r-th[i])) + K2/2 * (np.sin(2*th_r-th_l-th[i])+np.sin(2*th_l-th_r-th[i])))
    return new_th

#Movement is done by the velocity in the direction of the oscillator angle
def move_adim(r_v, th): #Done
    new_r_v = np.copy(r_v)
    for i in range(len(r_v)):
        new_r_v[i] += [dt_adim*np.cos(th[i]), dt_adim*np.sin(th[i])]

    return periodic_bc(new_r_v)

#Neighbors are two closest nodes
def kuramoto_neigh_adim(th, neigh): #Done
    new_th = np.copy(th)
    for i in range(len(th)):
        th_l = th[int(neigh[i,0])]
        th_r = th[int(neigh[i,1])]
        new_th[i] += dt_adim*(K12*(np.sin(th_l-th[i]) + np.sin(th_r-th[i])) + (np.sin(2*th_r-th_l-th[i])+np.sin(2*th_l-th_r-th[i])))
    return new_th

#All-to-all interactions
def kuramoto_global(th): #Done
    new_th = np.copy(th)
    for i in range(N):
        pairwise = np.sum(np.sin(th - th[i]))
        pairwise *= K1 / N
        threewise = 0
        for j in range(N):
            for l in range(N):
                if j != i and l != i:
                    threewise += np.sin(2 * th[j] - th[l] - th[i]) + np.sin(2 * th[l] - th[j] - th[i])
        threewise *= K2 / N**2
        new_th[i] += dt * (pairwise + threewise)
    return new_th

#Implements periodic boundary conditions
def periodic_bc(r_new): #Done
    return r_new % Lx

#Initializes N particles with N oscillators
def initialize_r0(): #Done
    r0_v = np.random.rand(N,2)*Lx #Create two random numbers between [0,1)
    th_0 = np.random.uniform(-np.pi, np.pi, N)
    return r0_v, th_0

#Find two closest particles of each particle
def find_neighbors(r_v):
    r_vt = np.copy(r_v)
    neigh = np.zeros((len(r_v),2))
    temp = np.zeros(len(r_v))
    for i in range(N):
        dr = r_v - r_v[i]
        dr -= Lx * np.round(dr / Lx)

        distances = np.linalg.norm(dr, axis=1)
        distances[i] = np.inf  # Ignore self

        nearest = np.argpartition(distances, 2)[:2]
        nearest = nearest[np.argsort(distances[nearest])]

        neigh[i] = nearest
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
        th_0 = kuramoto_neigh(th_0,neigh)
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
    plt.suptitle(f"$K_1$ = {K1}, $K_2$ = {K2}, v = {v}")

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        time = frame * dt
        sc.set_offsets(pos_mat[:, :, frame])
        sc.set_array(np.sin(act_mat[:, frame]))
        time_text.set_text(f'Time = {time:.2f}')
        return sc, time_text

    skip = 2  # Use every 3rd frame
    frames = range(0, act_mat.shape[1], skip)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True)

    if filename is not None:
        writervideo = animation.FFMpegWriter(fps=60)
        ani.save(filename = "../"+filename + "anim.mp4", writer = writervideo)
    plt.show()


#Plots the first, middle and last frame of the simulation
def plot_frames_from_data(filename = None):
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

    plt.show()


N= 100#number of particles
M = 1 #number of iterations to average

Lx = 5 #Box size
K1_v = [1,0,1,3,1,1,1] #Pairwise interaction parameter
K2_v = [0,1,1,1,3,10,100] #Trio-wise interaction parameter
K1 = 1.0
K2 = 5
K12 = K1/K2
K12_vec = np.linspace(0.01,3,10)
R1_K_mean = np.zeros((len(K12_vec), M))
R2_K_mean = np.zeros((len(K12_vec), M))
R3_K_mean = np.zeros((len(K12_vec), M))

R1_K = np.zeros(len(K12_vec))
R2_K = np.zeros(len(K12_vec))
R3_K = np.zeros(len(K12_vec))

T = 300
dt = 0.01 #Integration step
dt_adim = K2/2*dt
T_adim = K2/2*T
v = 1#Internal particle velocity
gamma = 10
t = np.linspace(0,T,int(T/dt))
t_adim = np.linspace(0,T_adim,int(T_adim/dt_adim))

#kplt.animate_oscillators(act_mat)
r_tilde_0, th_tilde_0 = initialize_r0()
#th_0 = [2*np.pi/3, 2*np.pi/3, -2*np.pi/3]
neigh = find_neighbors(r_tilde_0)
act_mat = np.reshape(th_tilde_0, (N,1))
pos_mat = np.zeros((N, 2, len(t)+1))
pos_mat[:, :, 0] = r_tilde_0

_, Kax = plt.subplots()
normalize = colors.Normalize(vmin=np.min(K12_vec), vmax=np.max(K12_vec))
cmap = cm.magma

for k in range(len(K12_vec)):
    K12 = K12_vec[k]
    dt_adim = (K1/(K12*2))*dt
    T_adim = (K1/(K12*2))*T
    t_adim = np.linspace(0,T_adim,int(T_adim/dt_adim))
    print(k)
    for l in range(M):
        r_tilde_0, th_tilde_0 = initialize_r0()
        neigh = find_neighbors(r_tilde_0)
        act_mat = np.reshape(th_tilde_0, (N,1))
        pos_mat = np.zeros((N, 2, len(t_adim)+1))
        pos_mat[:, :, 0] = r_tilde_0
        for i,_ in enumerate(t_adim):
            th_tilde_0 = kuramoto_neigh_adim(th_tilde_0, neigh)
            r_tilde_0 = move_adim(r_tilde_0,th_tilde_0)
            neigh = find_neighbors(r_tilde_0)
            pos_mat[:, :, i+1] = r_tilde_0
            act_mat = np.column_stack((act_mat,th_tilde_0))
        #R1_K_mean[k, l], R2_K_mean[k,l], R3_K_mean[k,l] = kplt.return_phase_mean(act_mat)
        color = cmap(normalize(K12))
        Kax.plot([kplt.phase_coherence(vec) for vec in act_mat.T], '.', label = f"K = {K12:.2f}", color = color)

    #R1_K[k] = np.mean(R1_K_mean[k])
    #R2_K[k] = np.mean(R2_K_mean[k])
    #R3_K[k] = np.mean(R3_K_mean[k])

    filename = f"K_{K12:.2f}"
    kplt.plot_phase_coherence_pair_three(act_mat,filename)
    plt.close()
    #plot_frames_from_data()

Kax.set_title(r"Order Parameter $R_1(t)$ for different $K = \frac{K_1}{K_2}$")
sm = cm.ScalarMappable(cmap=cmap, norm=normalize)
cbar = plt.colorbar(sm, ax=Kax)
cbar.set_label(r"$K = \frac{K_1}{K_2}$", rotation = 90,  labelpad=15)
plt.show()

_, ax = plt.subplots()
plt.plot(K12_vec, R1_K, label = "$R_1(K)$" )
plt.plot(K12_vec, R2_K, label = "$R_2(K)$" )
plt.plot(K12_vec, R3_K, label = "$R_3(K)$" )
plt.xlabel(r"$K = \frac{K_1}{K_2}$")
plt.ylabel(r"Order Parameters $<R_i>$")
plt.legend()
plt.show()

'''r_0, th_0 = initialize_r0()
neigh = find_neighbors(r_0)
act_mat = np.reshape(th_0, (N,1))
pos_mat = np.zeros((N, 2, len(t)+1))
pos_mat[:, :, 0] = r_0

for i,_ in enumerate(t_adim):
    th_0 = kuramoto_neigh_adim(th_0, neigh)
    r_0 = move_adim(r_0,th_0)
    neigh = find_neighbors(r_0)
    pos_mat[:, :, i+1] = r_0
    act_mat = np.column_stack((act_mat,th_0))

filename = "K1_"+str(K1)+"_K2_"+str(K2)+"_v_"+str(v)
kplt.plot_phase_coherence_pair_three(act_mat, filename)
animate_from_data(filename)
plot_frames_from_data(filename)'''

'''for j in range(len(K1_v)):
    K1 = K1_v[j]
    K2 = K2_v[j]

    r_0, th_0 = initialize_r0()
    neigh = find_neighbors(r_0)
    act_mat = np.reshape(th_0, (N,1))
    pos_mat = np.zeros((N, 2, len(t)+1))
    pos_mat[:, :, 0] = r_0

    for i,_ in enumerate(t):
        th_0 = kuramoto_neigh(th_0, neigh)
        r_0 = move(r_0,th_0)
        neigh = find_neighbors(r_0)
        pos_mat[:, :, i+1] = r_0
        act_mat = np.column_stack((act_mat,th_0))

    filename = "K1_"+str(K1)+"_K2_"+str(K2)+"_v_"+str(v)
    kplt.plot_phase_coherence_pair_three(act_mat, filename)
    animate_from_data(filename)
    plot_from_data(filename)'''

#kplt.plot_phase_coherence(act_mat)
#kplt.animate_oscillators(act_mat)
#kplt.plot_activity(act_mat)
#filename = "K1_"+str(K1)+"_K2_"+str(K2)+"_v_"+str(v)
#kplt.plot_phase_coherence_pair_three(act_mat, filename)
#animate_from_data(filename)
#plot_from_data(filename)

#animate()
