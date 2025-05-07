import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from kuramoto import Kuramoto


def plot_activity(act_mat, filename=None):
    """
    Plot sin(angle) vs time for each oscillator time series.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.sin(act_mat.T))
    ax.set_xlabel('Time', fontsize=15)
    ax.set_ylabel(r'$\sin(\theta)$', fontsize=15)
    #ax.set_title(f'Coupling = {coupling}' )
    if filename is not None:
        plt.savefig("../activity_coupling_"+filename+".png")

    plt.show()

    return ax


def plot_phase_coherence(act_mat, filename=None):
    """
    Plot order parameter phase_coherence vs time.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    _, ax = plt.subplots(figsize=(8, 3))
    ax.plot([Kuramoto.phase_coherence(vec) for vec in act_mat.T], 'o')
    ax.set_ylabel('Order parameter', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim((-0.01, 1))
    #ax.set_title(f'Coupling = {coupling}' )

    if filename is not None:
        plt.savefig("../phasecoherence_coupling_"+filename+".png")

    plt.show()

    return ax

# Plot oscillators in complex plane at times t = 0, 250, 500
def oscillators_comp(act_mat):
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5),
                         subplot_kw={
                             "ylim": (-1.1, 1.1),
                             "xlim": (-1.1, 1.1),
                             "xlabel": r'$\cos(\theta)$',
                             "ylabel": r'$\sin(\theta)$',
                         })

    times = [0, int(len(act_mat[0])/2), int(len(act_mat[0])-1)]
    print()
    for ax, time in zip(axes, times):
        print(times)
        ax.plot(np.cos(act_mat[:, time]),
                np.sin(act_mat[:, time]),
                'o',
                markersize=10)
        ax.set_title(f'Time = {time}')

    plt.show()

def animate_oscillators(act_mat, filename=None, frame_jump=30):

    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw={
                               "ylim": (-1.1, 1.1),
                               "xlim": (-1.1, 1.1),
                               "xlabel": r'$\cos(\theta)$',
                               "ylabel": r'$\sin(\theta)$',
                           })

    # Initialize the scatter plot
    dots, = ax.plot([], [], 'o', markersize=10)

    # Add time counter text (position is relative to plot coordinates)
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

    plt.show()
    plt.close()
