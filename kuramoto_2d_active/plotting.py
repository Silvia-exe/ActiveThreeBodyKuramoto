import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

'''Auxiliary library for actkurpy, for plotting and getting order parameters from act_mat and pos_mat'''

#Compute global order parameter R_t
def phase_coherence(angles_vec):
        suma = sum([(np.e ** (1j * i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

#Compute global order parameter R_t
def phase_coherence_pair(angles_vec):
        suma = sum([(np.e ** (1j * 2*i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

#Compute global order parameter R_t
def phase_coherence_three(angles_vec):
        suma = sum([(np.e ** (1j * 3*i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

#Plot sin(angle) as a function of time for each oscillator.
def plot_activity(act_mat, filename=None):
    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.sin(act_mat.T))
    ax.set_xlabel('Time', fontsize=15)
    ax.set_ylabel(r'$\sin(\theta)$', fontsize=15)
    #ax.set_title(f'Coupling = {coupling}' )
    if filename is not None:
        plt.savefig("../activity_coupling_"+filename+".png")

    plt.show()

    return ax

#Plot order parameter R1 of system as a function of time.
def plot_phase_coherence(act_mat, title, show = True, filename=None):
    _, ax = plt.subplots()#figsize=(8, 4))
    ax.plot([phase_coherence(vec) for vec in act_mat.T], 'o')
    ax.set_ylabel(r'R($\theta$)', fontsize=15)
    ax.set_xlabel('Time [t]', fontsize=15)
    plt.suptitle(title)
    if filename is not None:
        ax.set_title(filename)
        plt.savefig("../phasecoherence_coupling_"+filename+".png")
    if show == True:
        plt.show()
    return ax

#Plot order parameter of system as a function of time.
def plot_phase_coherence_pair_three(act_mat, title, show = True, filename=None):
    _, ax = plt.subplots()#figsize=(8, 4))
    ax.plot([phase_coherence(vec) for vec in act_mat.T], label = "$R_1$")
    ax.plot([phase_coherence_pair(vec) for vec in act_mat.T], label = "$R_2$")
    ax.plot([phase_coherence_three(vec) for vec in act_mat.T], label = "$R_3$")
    ax.set_ylabel(r'$R_n(\theta)$', fontsize=15)
    ax.set_xlabel('Time [t]', fontsize=15)
    ax.legend()
    plt.suptitle(title)
    if filename is not None:
        plt.savefig("../phasecoherence_k2k3_"+filename+".png")
    if show== True:
        plt.show()
    return ax

#Returns the mean of the phase coherence R1, R2 and R3 for the last 10% of the points
def return_phase_mean(act_mat):
    R1_mean = np.mean(np.array([phase_coherence(vec) for vec in act_mat.T])[int(-0.1*act_mat.shape[1]):])
    R2_mean = np.mean(np.array([phase_coherence_pair(vec) for vec in act_mat.T])[int(-0.1*act_mat.shape[1]):])
    R3_mean = np.mean(np.array([phase_coherence_three(vec) for vec in act_mat.T])[int(-0.1*act_mat.shape[1]):])

    return (R1_mean, R2_mean, R3_mean)

#Returns the standard deviation of the phase coherence R1, R2 and R3 for the last 10% of the points
def return_phase_std(act_mat):
    R1_std = np.std(np.array([phase_coherence(vec) for vec in act_mat.T])[int(-0.1*act_mat.shape[1]):])
    R2_std = np.std(np.array([phase_coherence_pair(vec) for vec in act_mat.T])[int(-0.1*act_mat.shape[1]):])
    R3_std = np.std(np.array([phase_coherence_three(vec) for vec in act_mat.T])[int(-0.1*act_mat.shape[1]):])

    return (R1_std, R2_std, R3_std)

# Plot oscillators in complex plane at times t = 0, T/2, T
def oscillators_comp(act_mat):
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5),
                         subplot_kw={
                             "ylim": (-1.1, 1.1),
                             "xlim": (-1.1, 1.1),
                             "xlabel": r'$\cos(\theta)$',
                             "ylabel": r'$\sin(\theta)$',
                         })

    times = [0, int(len(act_mat[0])/2), -1]
    for ax, time in zip(axes, times):
        ax.plot(np.cos(act_mat[:, time]),
                np.sin(act_mat[:, time]),
                'o',
                markersize=10)
        ax.set_title(f'Time = {time}')

    #plt.suptitle(r'$K_1 = 1, K_2 = 3,\alpha = 2\pi/3$ ', size = "x-large")
    plt.show()

#Animates oscillators in a unit circle (cos(theta) vs sin(theta))
def animate_oscillators(act_mat, show =True, filename=None, frame_jump=30):

    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw={
                               "ylim": (-1.1, 1.1),
                               "xlim": (-1.1, 1.1),
                               "xlabel": r'$\cos(\theta)$',
                               "ylabel": r'$\sin(\theta)$',
                           })

    dots, = ax.plot([], [], 'o', markersize=10)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')
    def init():
        dots.set_data([], [])
        return dots,
    def animate(i):
        x = np.cos(act_mat[:, i])
        y = np.sin(act_mat[:, i])
        dots.set_data(x, y)
        time_text.set_text(f'Time = {i}')
        return dots,
    ani = animation.FuncAnimation(
        fig, animate, frames=act_mat.shape[1],
        init_func=init, interval=frame_jump,
        blit=False, repeat=False)
    if filename is not None:
        ani.save(filename = "../"+filename + ".gif", writer = "pillow")
    if show == True:
        plt.show()

#For animating the produced act_mat and pos_mat of 2D simulation
def animate_active_oscillators(act_mat, pos_mat, title, Lx = 5, dt = 0.01, show = True, filename = None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_xlim(0, Lx)
    ax.set_aspect('equal', adjustable='box')

    normalize = colors.Normalize(vmin=-1, vmax=1)
    sc = ax.scatter(pos_mat[:, 0, 0], pos_mat[:, 1, 0], c=np.sin(act_mat[:, 0]), norm= normalize, cmap="viridis", edgecolor="black", s = 100)
    cb = plt.colorbar(sc)
    cb.set_label(r'sin($\theta$)')
    plt.suptitle(title)

    time_text = ax.text(0.017, 0.952, '', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha = 0.8))
    R1_text = ax.text(0.017, 0.765, '', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', alpha = 0.8))

    def update(frame):
        time = frame * dt
        sc.set_offsets(pos_mat[:, :, frame])
        sc.set_array(np.sin(act_mat[:, frame]))
        time_text.set_text(f'Time = {time:.2f}')
        R1_text.set_text(f'$R_1$ = {phase_coherence(act_mat[:, frame]):.2f}\n$R_2$ = {phase_coherence_pair(act_mat[:, frame]):.2f}\n$R_3$ = {phase_coherence_three(act_mat[:, frame]):.2f}')
        return sc, time_text, R1_text

    skip = 2  # Use every 3rd frame
    frames = range(0, act_mat.shape[1], skip)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True)

    if filename is not None:
        writervideo = animation.FFMpegWriter(fps=60)
        ani.save(filename = "../"+filename + "anim.mp4", writer = writervideo)

    if show == True:
        plt.show()

#Plots the first, middle and last frame of the simulation
def plot_frames_from_data(act_mat, pos_mat, title, Lx = 5, show = True,  filename = None):
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
    plt.suptitle(title)
    if filename is not None:
        plt.savefig("../"+filename+"_seq.png")
    if show == True:
        plt.show()
