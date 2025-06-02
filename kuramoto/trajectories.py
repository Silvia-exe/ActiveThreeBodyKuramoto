import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Patch

#Global parameters
w1, w2, w3 = 0.0, 0.0, 0.0 #Natural frequencies
K1, K2 = 1.0, 2.0
unit   = 1/3 #For labelling every third of pi
y_tick = np.arange(-1, 1+unit, unit)
y_label = [r"$-\pi$",r"$-2\frac{\pi}{3}$" ,r"$-\frac{\pi}{3}$", r"$0$", r"$\frac{\pi}{3}$",   r"$2\frac{\pi}{3}$",r"$\pi$"]

N = 30 #Number of angles to sample
phi2_vals = np.linspace(-np.pi, np.pi, N)
phi3_vals = np.linspace(-np.pi, np.pi, N)
phi2, phi3 = np.meshgrid(phi2_vals, phi3_vals)

epsilon = 1e-10 #Small threshold

N_init = 10

#N3 Kuramoto for a reduced frame of reference system
def N3_Kuramoto(t, y):
    phi2, phi3 = y
    dphi2 = w2 - w1 + K1/2 * (np.sin(phi3 - phi2) - 2*np.sin(phi2) - np.sin(phi3)) + K2/2 * (np.sin(phi3 - 2*phi2) - np.sin(phi2 + phi3))
    dphi3 = w3 - w1 + K1/2 * (np.sin(phi2 - phi3) - 2*np.sin(phi3) - np.sin(phi2)) + K2/2 * (np.sin(phi2 - 2*phi3) - np.sin(phi2 + phi3))
    return [dphi2, dphi3]

#Computes vector field of dphi2 and dphi3 and its "speed"
def compute_vector_field(N=25):

    Phi2, Phi3 = np.meshgrid(phi2_vals, phi3_vals)
    U = np.zeros_like(Phi2)
    V = np.zeros_like(Phi3)

    for i in range(N):
        for j in range(N):
            dphi2, dphi3 = N3_Kuramoto_dphi_dt(0,[Phi[i], Phi[j]])
            U[i,j] = dphi2
            V[i,j] = dphi3

    speed = np.sqrt(dphi2**2 + dphi3**2) + epsilon
    U /= speed
    V /= speed
    return Phi2, Phi3, U, V, speed

#Plots vector field as a streamplot
def plot_vector_field(N=25):
    fig, ax = plt.subplots(figsize=(10, 8))
    Phi2, Phi3, U, V, speed = compute_vector_field(N)
    stream = ax.streamplot(Phi2, Phi3, U, V, color=speed, cmap='viridis', density=1.3, arrowsize=1.5)
    cbar = fig.colorbar(stream.lines, ax=ax)
    cbar.set_label('Speed')
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_xlabel(r'$\psi_2$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title('Phase flow field for $K_1=$' + str(K1) + " and $K_2=$"+str(K2))
    ax.set_yticks(y_tick*np.pi)
    ax.set_xticks(y_tick*np.pi)
    ax.set_yticklabels(y_label, fontsize=12)
    ax.set_xticklabels(y_label, fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True)
    plt.show()
    return stream

# Plots trajectories of each initial condition as arrows
def plot_trajectories(Tmax=50, skip=50):
    fig, ax = plt.subplots(figsize=(10, 8))
    t_eval = np.linspace(0, Tmax, 1000)
    for phi2_0 in phi2_vals:
        for phi3_0 in phi3_vals:
            sol = solve_ivp(N3_Kuramoto_dphi_dt, [0, Tmax], [phi2_0, phi3_0], t_eval=t_eval)
            phi2_traj, phi3_traj = sol.y

            ax.quiver(
                phi2_traj[:-skip:skip], phi3_traj[:-skip:skip],
                phi2_traj[skip::skip] - phi2_traj[:-skip:skip],
                phi3_traj[skip::skip] - phi3_traj[:-skip:skip],
                angles='xy', scale_units='xy', scale=1, color='blue', width=0.003, alpha=0.8
            )
            ax.scatter(phi2_traj[0], phi3_traj[0], color='green', s=10)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_xlabel(r'$\psi_2$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title('Trajectories for $K_1=$' + str(K1) + " and $K_2=$"+str(K2))
    ax.set_yticks(y_tick*np.pi)
    ax.set_xticks(y_tick*np.pi)
    ax.set_yticklabels(y_label, fontsize=12)
    ax.set_xticklabels(y_label, fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True)
    plt.show()

#Plots basins of attraction (the three stable ones)
def plot_basins(Tmax=50):

    color_grid = np.zeros((phi2_vals.shape[0], phi2_vals.shape[0], 3))

    attractors = [
        np.array([0, 0]),
        np.array([2 * np.pi / 3, -2*np.pi / 3]),
        np.array([-2*np.pi / 3, 2 * np.pi / 3])
    ]
    colors = [
        np.array([1, 0, 0]),  # Red
        np.array([0, 1, 0]),  # Green
        np.array([0, 0, 1])   # Blue
    ]

    for i in range(phi2.shape[0]):
        for j in range(phi2.shape[1]):
            phi2_0 = phi2[i, j]
            phi3_0 = phi3[i, j]
            sol = solve_ivp(N3_Kuramoto_dphi_dt, [0, Tmax], [phi2_0, phi3_0], t_eval=[Tmax])
            phi2_f, phi3_f = sol.y[:, -1]

            final_state = np.array([phi2_f, phi3_f])
            distances = [np.linalg.norm(final_state - a) for a in attractors]
            min_idx = np.argmin(distances)
            color_grid[i, j] = colors[min_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(color_grid, origin='lower',
                   extent=[-np.pi, np.pi, -np.pi, np.pi], aspect='equal')

    ax.set_xlabel(r'$\psi_2$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title(r'Basins of attraction of N = 3 for $K_1=$' + str(K1) + " and $K_2=$"+str(K2))
    ax.set_yticks(y_tick*np.pi)
    ax.set_xticks(y_tick*np.pi)
    ax.set_yticklabels(y_label, fontsize=12)
    ax.set_xticklabels(y_label, fontsize=12)
    ax.grid(True)

    legend_elements = [
        Patch(facecolor=colors[0], label='Attractor (0, 0)'),
        Patch(facecolor=colors[1], label=r'Attractor $(-\frac{2\pi}{3}, \frac{2\pi}{3})$'),
        Patch(facecolor=colors[2], label=r'Attractor $(\frac{2\pi}{3}, -\frac{2\pi}{3})$')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    for a, c in zip(attractors, colors):
        ax.plot(a[0], a[1], 'o', color=c, markersize=8, markeredgecolor='k')
    plt.show()


#plot_trajectories(Tmax=10)
#plot_vector_field()
plot_basins()
