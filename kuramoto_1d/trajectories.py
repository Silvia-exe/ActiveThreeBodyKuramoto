import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp

#Global parameters
w1, w2, w3 = 0.0, 0.0, 0.0 #Natural frequencies
K1, K2 = 1.0, 2.0
K12 = 0.0
unit   = 1/3 #For labelling every third of pi
y_tick = np.arange(-1, 1+unit, unit)
y_label = [r"$-\pi$",r"$-2\frac{\pi}{3}$" ,r"$-\frac{\pi}{3}$", r"$0$", r"$\frac{\pi}{3}$",   r"$2\frac{\pi}{3}$",r"$\pi$"]
y_label = [r"$-\pi$",r"$-2\frac{\pi}{3}$" ,r"$-\frac{\pi}{3}$", r"$0$", r"$\frac{\pi}{3}$",   r"$2\frac{\pi}{3}$",r"$\pi$"]

N = 303 #Number of angles to sample
phi_frac = np.array([np.pi,np.pi/2,np.pi/3,2*np.pi/3, np.pi/4, 3*np.pi/4, np.pi/6, 5*np.pi/6]) #Some angles of interest that must be sampled

phi2_vals = np.linspace(-np.pi, np.pi, N)
phi2_vals = np.append(phi2_vals, [phi_frac,-phi_frac])
phi2_vals = np.unique(np.sort(phi2_vals),axis = 0)

phi3_vals = np.linspace(-np.pi, np.pi, N)
phi3_vals = np.append(phi3_vals, [phi_frac,-phi_frac])
phi3_vals = np.unique(np.sort(phi3_vals),axis = 0)

phi2, phi3 = np.meshgrid(phi2_vals, phi3_vals)

epsilon = 1e-10 #Small threshold

N_init = 10

#N3 Kuramoto for a reduced frame of reference system
def N3_Kuramoto(t, y):
    phi2, phi3 = y
    dphi2 = w2 - w1 + K1/2 * (np.sin(phi3 - phi2) - 2*np.sin(phi2) - np.sin(phi3)) + K2/2 * (np.sin(phi3 - 2*phi2) - np.sin(phi2 + phi3))
    dphi3 = w3 - w1 + K1/2 * (np.sin(phi2 - phi3) - 2*np.sin(phi3) - np.sin(phi2)) + K2/2 * (np.sin(phi2 - 2*phi3) - np.sin(phi2 + phi3))
    return [dphi2, dphi3]

#N3 Kuramoto for a reduced frame of reference and adimensionalized
def adim_N3_Kuramoto(t, y):
    phi2, phi3 = y
    dphi2 = w2 - w1 + K12 * (np.sin(phi3-phi2)-2*np.sin(phi2)-np.sin(phi3)) + (np.sin(phi3-2*phi2)-np.sin(phi2+phi3))
    dphi3 = w3 - w1 + K12 * (np.sin(phi2-phi3)-2*np.sin(phi3)-np.sin(phi2)) + (np.sin(phi2-2*phi3)-np.sin(phi3+phi2))
    return [dphi2, dphi3]

#Computes vector field of dphi2 and dphi3 and its "speed"
def compute_vector_field():
    U = np.zeros_like(phi2)
    V = np.zeros_like(phi3)

    for i in range(N):
        for j in range(N):
            dphi2, dphi3 = adim_N3_Kuramoto(0,[phi2[i,j], phi3[i,j]])
            U[i,j] = dphi2
            V[i,j] = dphi3

    speed = np.sqrt(U**2 + V**2) + epsilon
    U /= speed
    V /= speed
    return phi2, phi3, U, V, speed

#Plots vector field as a streamplot
def plot_vector_field(N=25):
    fig, ax = plt.subplots(figsize=(10, 8))
    Phi2, Phi3, U, V, speed = compute_vector_field()
    stream = ax.streamplot(Phi2, Phi3, U, V, color=speed, cmap='viridis', density=1.3, arrowsize=1.5)
    cbar = fig.colorbar(stream.lines, ax=ax)
    cbar.set_label('Speed')
    #ax.set_xlim([-np.pi, np.pi])
    #ax.set_ylim([-np.pi, np.pi])
    ax.set_xlabel(r'$\psi_2$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title('Phase flow field for $K=$' + str(K12))
    #ax.set_yticks(y_tick*np.pi)
    #ax.set_xticks(y_tick*np.pi)
    #ax.set_yticklabels(y_label, fontsize=12)
    #ax.set_xticklabels(y_label, fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True)
    plt.show()
    return stream

# Plots trajectories of each initial condition as arrows
def plot_trajectories(Tmax=100, skip=50):
    N= len(phi2_vals) #Updating the number of angles to sample
    fig, ax = plt.subplots(figsize=(9, 8))
    t_eval = np.linspace(0, Tmax, 3500) #Time vector to integrate
    final_states = []
    for phi2_0 in phi2_vals:
        for phi3_0 in phi3_vals:
            sol = solve_ivp(adim_N3_Kuramoto, [0, Tmax], [phi2_0, phi3_0], t_eval=t_eval, method = 'Radau') #Getting trajectories with Radau (for stiff equations)
            phi2_traj, phi3_traj = sol.y
            '''ax.quiver(
                phi2_traj[:-skip:skip], phi3_traj[:-skip:skip],
                phi2_traj[skip::skip] - phi2_traj[:-skip:skip],
                phi3_traj[skip::skip] - phi3_traj[:-skip:skip],
                angles='xy', scale_units='xy', scale=1, color='gray', width=0.008, alpha=0.4
            )'''
            phi2_final, phi3_final = sol.y[0, -1], sol.y[1, -1]
            final_states.append([(phi2_final+np.pi)%(2*np.pi)-np.pi, (phi3_final+np.pi)%(2*np.pi)-np.pi])

    #Getting all the unique final states (rounded to one decimal), and the absolute value.
    #final_state_inv is the index of the fixed point that angle pair tended to.
    final_states_unique= np.unique((np.array(final_states).round(1)),axis=0)
    final_states_unique_abs, final_state_inv  = np.unique(np.sort(abs(np.array(final_states)).round(1)),axis=0,return_inverse = True)

    print(final_states_unique)
    final_state_inv = (np.reshape(final_state_inv,(N,N)))

    #Making labels from the unique fixed points
    labels = make_labels(final_states_unique)

    num_states = len(final_states_unique_abs)
    base_cmap = plt.get_cmap('tab10')
    colors = base_cmap.colors[:num_states]
    cmap = ListedColormap(colors)
    norm = Normalize(vmin=0, vmax=num_states - 1)

    im = ax.imshow(final_state_inv, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=cmap,norm = norm, interpolation='nearest')
    legend_elements = [Patch(facecolor=cmap(i), label=labels[i]) for i in np.unique(final_state_inv)]
    lgd = ax.legend(handles=legend_elements, bbox_to_anchor= (1.4, 0.65), title = "Fixed Points")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_xlabel(r'$\psi_2$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title('Basins of attraction for $K=$' + str(K12))
    ax.set_yticks(y_tick*np.pi)
    ax.set_xticks(y_tick*np.pi)
    ax.set_yticklabels(y_label, fontsize=12)
    ax.set_xticklabels(y_label, fontsize=12)
    plt.tight_layout()
    ax.grid(True)

    fig.savefig(f'basins_N_{N}_K_{K12:.2f}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')

    with h5py.File(f'basins_N_{N}_K_{K12:.2f}.h5','w') as h5f:
        h5f.create_group('parameters')

        para_list = []
        para_list += [("/parameters/K", K12)]
        para_list += [("/parameters/T", Tmax)]

        for element in para_list:
            h5f.create_dataset(element[0],data=element[1])

        h5f.create_group('fp_array_inv')
        h5f.create_group('fp_list')
        h5f.create_group('fp_list_abs')

        h5f.create_dataset("fp_array_inv/fp_array", data = final_state_inv, compression= "gzip", dtype = np.int32, chunks = True, compression_opts = 6)
        h5f.create_dataset("fp_list/fp_list", data = final_states_unique, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)
        h5f.create_dataset("fp_list_abs/fp_list_abs", data = final_states_unique_abs, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)

    plt.show()

#Creates labels from final states, for plotting
def make_labels(final_states):
    states = np.round(final_states, 1)
    final_states_abs = np.unique(np.round(np.abs(states), 1), axis=0)

    def to_angle(val):
        mapping = {
            0.0: "0",
            1.0: "π/3",
            1.6: "π/2",
            2.1: "2π/3",
            3.1: "π",
        }
        val = np.round(abs(val), 1)
        return mapping.get(val, f"{val:.1f}")

    abs_states = np.unique(np.round(np.abs(states), 1), axis=0)

    labels = []

    for i, (phi2, phi3) in enumerate(abs_states, 1):
        mask = (np.round(np.abs(states[:, 0]), 1) == phi2) & \
               (np.round(np.abs(states[:, 1]), 1) == phi3)
        subset = states[mask]

        signs = np.sign(subset)
        sign_products = np.unique(signs[:, 0] * signs[:, 1])

        phi2_str = to_angle(phi2)
        phi3_str = to_angle(phi3)

        phi2_label = f"±{phi2_str}" if phi2 != 0 else phi2_str
        phi3_label = f"±{phi3_str}" if phi3 != 0 else phi3_str

        if len(sign_products) == 1:
            if sign_products[0] == 1:
                phi2_label = f"±{phi2_str}"
                phi3_label = f"±{phi3_str}"
            elif sign_products[0] == -1:
                phi2_label = f"±{phi2_str}"
                phi3_label = f"∓{phi3_str}"
        else:
            phi2_label = f"±{phi2_str}" if phi2 != 0 else phi2_str
            phi3_label = f"±{phi3_str}" if phi3 != 0 else phi3_str

        labels.append(fr'{i}: $\psi_2=$ {phi2_label}, $\psi_3=$ {phi3_label}')

    return labels

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
            sol = solve_ivp(adim_N3_Kuramoto, [0, Tmax], [phi2_0, phi3_0], t_eval=[Tmax])
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


plot_trajectories()
#plot_vector_field()
#plot_basins()
