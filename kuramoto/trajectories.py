import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Global Parameters ---
w1, w2, w3 = 0.0, 0.0, 0.0
K1, K2 = 1.0, 5.0
unit   = 1/3
y_tick = np.arange(-1, 1+unit, unit)
y_label = [r"$-\pi$",r"$-2\frac{\pi}{3}$" ,r"$-\frac{\pi}{3}$", r"$0$", r"$\frac{\pi}{3}$",   r"$2\frac{\pi}{3}$",r"$\pi$"]

# --- Phase Dynamics ODE ---
def N3_Kuramoto_dphi_dt(t, y):
    phi2, phi3 = y
    dphi2 = w2 - w1 + K1/2 * (np.sin(phi3 - phi2) - 2*np.sin(phi2) - np.sin(phi3)) \
                    + K2/2 * (np.sin(phi3 - 2*phi2) - np.sin(phi2 + phi3))
    dphi3 = w3 - w1 + K1/2 * (np.sin(phi2 - phi3) - 2*np.sin(phi3) - np.sin(phi2)) \
                    + K2/2 * (np.sin(phi2 - 2*phi3) - np.sin(phi2 + phi3))
    return [dphi2, dphi3]

# --- Compute Vector Field ---
def compute_vector_field(N=25):
    phi2_vals = np.linspace(-np.pi, np.pi, N)
    phi3_vals = np.linspace(-np.pi, np.pi, N)
    Phi2, Phi3 = np.meshgrid(phi2_vals, phi3_vals)

    dphi2 = w2 - w1 + K1/2 * (np.sin(Phi3 - Phi2) - 2*np.sin(Phi2) - np.sin(Phi3)) \
                    + K2/2 * (np.sin(Phi3 - 2*Phi2) - np.sin(Phi2 + Phi3))
    dphi3 = w3 - w1 + K1/2 * (np.sin(Phi2 - Phi3) - 2*np.sin(Phi3) - np.sin(Phi2)) \
                    + K2/2 * (np.sin(Phi2 - 2*Phi3) - np.sin(Phi2 + Phi3))

    epsilon = 1e-10
    speed = np.sqrt(dphi2**2 + dphi3**2) + epsilon
    U = dphi2 / speed
    V = dphi3 / speed
    return Phi2, Phi3, U, V, speed

# --- Plot Vector Field as Streamplot ---
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

# --- Plot Trajectories with Arrows ---
def plot_trajectories(phi2_vals, phi3_vals, Tmax=50, skip=50):
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

# --- Main Function to Plot Everything ---
def main():

    N_init = 10
    phi2_vals = np.linspace(-np.pi, np.pi, N_init)
    phi3_vals = np.linspace(-np.pi, np.pi, N_init)
    plot_trajectories(phi2_vals, phi3_vals, Tmax=10)
    plot_vector_field()


if __name__ == '__main__':
    main()
