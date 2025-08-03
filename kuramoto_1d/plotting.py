import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

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

#Plot order parameter of system as a function of time.
def plot_phase_coherence(act_mat, filename=None):
    _, ax = plt.subplots(figsize=(8, 3))
    ax.plot([phase_coherence(vec) for vec in act_mat.T], 'o')
    ax.set_ylabel('Order parameter', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim((-0.01, 1))
    #ax.set_title(f'Coupling = {coupling}' )

    if filename is not None:
        plt.savefig("../phasecoherence_coupling_"+filename+".png")

    #plt.show()

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
        plt.savefig("../order_par_"+filename+".png")
    if show== True:
        plt.show()
    return ax

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

    plt.suptitle(r'$K_1 = 1, K_2 = 3,\alpha = 2\pi/3$ ', size = "x-large")
    plt.show()

#Animates oscillators in a unit circle (cos(theta) vs sin(theta))
def animate_oscillators(act_mat,title, show = True, filename=None, frame_jump=30):

    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw={
                               "ylim": (-1.1, 1.1),
                               "xlim": (-1.1, 1.1),
                               "xlabel": r'$\cos(\theta)$',
                               "ylabel": r'$\sin(\theta)$',
                           })
    ax.set_title(title)

    N = act_mat.shape[0]
    even_idx = np.arange(0, N, 2)
    odd_idx  = np.arange(1, N, 2)

    # Two groups of scatter plots
    dots_even, = ax.plot([], [], 'o', markersize=15, fillstyle='none', color='blue')
    dots_odd,  = ax.plot([], [], 'o', markersize=10, color='blue')

    # Time label
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')

    plt.tight_layout()
    def init():
        dots_even.set_data([], [])
        dots_odd.set_data([], [])
        time_text.set_text('')
        return dots_even, dots_odd, time_text

    def animate(i):
        x = np.cos(act_mat[:, i])
        y = np.sin(act_mat[:, i])
        dots_even.set_data(x[even_idx], y[even_idx])
        dots_odd.set_data(x[odd_idx], y[odd_idx])
        time_text.set_text(f'Time = {i}')
        return dots_even, dots_odd, time_text

    ani = animation.FuncAnimation(
        fig, animate, frames=act_mat.shape[1],
        init_func=init, interval=frame_jump,
        blit=True, repeat=False)

    if filename is not None:
        ani.save("../" + filename + ".gif", writer="pillow")

    if show == True:
        plt.show()
        plt.close()
