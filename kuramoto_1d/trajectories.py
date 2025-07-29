import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sympy as sp
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from scipy.optimize import root

#Defining symbols for Jacobian, to analyze Fixed Point stability
w = sp.symbols('w1:5') # w1 to w4
K = sp.symbols('K') #Coupling parameter
sphi2, sphi3 = sp.symbols('sphi2 sphi3') #Symbols for Jacobian
lam = sp.Symbol('lambda') #Lambda symbol for eigenvalues
variables_movfr = [sphi2, sphi3] #ph2 = th2-th1 and ph3 = th3-th1

#Moving frame for adimensional Kuramoto model, 1D, 3-node system
f_movfr_N3_adim = [
    w[2] - w[1] + K*(sp.sin(sphi3-sphi2)-2*sp.sin(sphi2)-sp.sin(sphi3)) + (sp.sin(sphi3-2*sphi2)-sp.sin(sphi2+sphi3)),
    w[3] - w[1] + K*(sp.sin(sphi2-sphi3)-2*sp.sin(sphi3)-sp.sin(sphi2)) + (sp.sin(sphi2-2*sphi3)-sp.sin(sphi3+sphi2))
]

#Global parameters
w1, w2, w3 = 0.0, 0.0, 0.0 #Natural frequencies
K1, K2 = 1.0, 2.0 #Coupling parameters for pairwise and threewise interactions
K12 = 0.2 #Coupling parameter for adimensionalized model K = K1/K2

#Plotting parameters and sampling
unit   = 1/3 #For labelling every third of pi
axis_tick = np.arange(-1, 1+unit, unit) #For ticks of the acis
axis_label = [r"$-\pi$",r"$-2\frac{\pi}{3}$" ,r"$-\frac{\pi}{3}$", r"$0$", r"$\frac{\pi}{3}$",   r"$2\frac{\pi}{3}$",r"$\pi$"]

N = 113 #Number of angles to sample
phi_frac = np.array([np.pi,np.pi/2,np.pi/3,2*np.pi/3, np.pi/4, 3*np.pi/4, np.pi/6, 5*np.pi/6]) #Some angles of interest that must be sampled

phi2_vals = np.linspace(-np.pi-np.pi/8, np.pi+np.pi/8, N)
phi2_vals = np.append(phi2_vals, [phi_frac,-phi_frac])
phi2_vals = np.unique(np.sort(phi2_vals),axis = 0)

phi3_vals = np.linspace(-np.pi-np.pi/8, np.pi+np.pi/8, N)
phi3_vals = np.append(phi3_vals, [phi_frac,-phi_frac])
phi3_vals = np.unique(np.sort(phi3_vals),axis = 0)

phi2, phi3 = np.meshgrid(phi2_vals, phi3_vals)

epsilon = 1e-3 #Small threshold

#N3 Kuramoto for a reduced frame of reference system
def N3_Kuramoto(t, y):
    phi2, phi3 = y
    dphi2 = w2 - w1 + K1/2 * (np.sin(phi3 - phi2) - 2*np.sin(phi2) - np.sin(phi3)) + K2/2 * (np.sin(phi3 - 2*phi2) - np.sin(phi2 + phi3))
    dphi3 = w3 - w1 + K1/2 * (np.sin(phi2 - phi3) - 2*np.sin(phi3) - np.sin(phi2)) + K2/2 * (np.sin(phi2 - 2*phi3) - np.sin(phi2 + phi3))
    return [dphi2, dphi3]

#N3 Kuramoto for a reduced frame of reference and adimensionalized
def adim_N3_Kuramoto(t, y, K):
    phi2, phi3 = y
    dphi2 = w2 - w1 + K * (np.sin(phi3-phi2)-2*np.sin(phi2)-np.sin(phi3)) + (np.sin(phi3-2*phi2)-np.sin(phi2+phi3))
    dphi3 = w3 - w1 + K * (np.sin(phi2-phi3)-2*np.sin(phi3)-np.sin(phi2)) + (np.sin(phi2-2*phi3)-np.sin(phi3+phi2))
    return [dphi2, dphi3]

#Plots nullclines and fixed points (nullcline intersections) with stability and type.
def plot_nullclines():
    def compute_nullclines(K):
        fixed_points = []
        Dphi2 = np.zeros_like(phi2)
        Dphi3 = np.zeros_like(phi3)
        for i in range(phi2.shape[0]):
            for j in range(phi2.shape[1]):
                dphi2_val, dphi3_val = adim_N3_Kuramoto(0, [phi2[i, j], phi3[i, j]], K)
                Dphi2[i,j] = dphi2_val
                Dphi3[i,j] = dphi3_val

        fixed_points = classify_fp(find_roots(K),K)

        return (Dphi2,Dphi3,fixed_points)

    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8),
                         subplot_kw={"xlabel": r'$\psi_2$', "ylabel": r'$\psi_3$'})
    K_vals = [0,0.15,0.3,0.6,1.0,1.5,2.0,3.0,4.0]
    axes = axes.flatten()
    markers = ['o', r'$\infty$', '*',r'$\asymp$',r'$\parallel$']
    colors = ['r', 'b']
    used_marker_styles = set()

    for i, (ax, k) in enumerate(zip(axes,K_vals)):
        Dphi2, Dphi3, fixed_points = compute_nullclines(k)
        ax.contour(phi2, phi3, Dphi2, levels=[0], colors='C0', linestyles='-')
        ax.contour(phi2, phi3, Dphi3, levels=[0], colors='orange', linestyles='-')

        for fp in fixed_points:
            psi2, psi3 = fp[0], fp[1]
            stability = int(fp[2])
            fp_type = int(fp[3])

            color = colors[stability]
            marker = markers[fp_type]

            ax.scatter(psi2, psi3, color=color, marker=marker, s=80)
            used_marker_styles.add((marker, color, stability, fp_type))

        ax.set_title(f"K={k:.2f}")
        ax.set_xlabel(r'$\psi_2$')
        ax.set_ylabel(r'$\psi_3$')
        ax.set_yticks(axis_tick*np.pi)
        ax.set_xticks(axis_tick*np.pi)
        ax.set_yticklabels(axis_label, fontsize=12)
        ax.set_xticklabels(axis_label, fontsize=12)
        ax.set_xlim(-np.pi-0.2, np.pi+0.2)
        ax.set_ylim(-np.pi-0.2, np.pi+0.2)
        ax.set_aspect('equal')
        ax.grid(True)

        if i < 6:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        if i % 3 != 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')

        if i == 5:
            nullcline_legend = [
                Line2D([0], [0], color='C0', linestyle='-', label=r'$d{\psi}_2 = 0$'),
                Line2D([0], [0], color='orange', linestyle='-', label=r'$d{\psi}_3 = 0$')
            ]
            label_map = {
                ('o', 0): 'Unstable Node',
                ('o', 1): 'Stable Node',
                (r'$\infty$', 0): 'Unstable Spiral',
                (r'$\infty$', 1): 'Stable Spiral',
                ('*', 0): 'Unstable Degenerate Node',
                ('*', 1): 'Stable Degenerate Node',
                (r'$\asymp$', 0): 'Saddle Point',
                (r'$\parallel$', 1): 'Center / Line'
            }

            marker_legend = []
            for marker, color, stability, fp_type in used_marker_styles:
                label = label_map.get((marker, stability), f"Type {fp_type}")
                marker_legend.append(Line2D([0], [0], marker=marker, color=color, label=label,
                                             linestyle='None', markersize=10))

            all_handles = nullcline_legend + marker_legend
            ax.legend(handles=all_handles,
              loc='center left',
              bbox_to_anchor=(1.05, 0.5),
              borderaxespad=0.,
              title="Fixed points")

    fig.suptitle("Nullclines")

    plt.show()

#Computes vector field of dphi2 and dphi3 and its "speed"
def compute_vector_field(K):
    fp_phi2 = np.pi/3
    fp_phi3 = -np.pi/3
    phi2_v = np.linspace(fp_phi2-np.pi/8, fp_phi2+np.pi/8, N)
    phi3_v = np.linspace(fp_phi3-np.pi/8, fp_phi3+np.pi/8, N)
    phi2, phi3 = np.meshgrid(phi2_v, phi3_v)
    U = np.zeros_like(phi2)
    V = np.zeros_like(phi3)

    for i in range(N):
        for j in range(N):
            dphi2, dphi3 = adim_N3_Kuramoto(0,[phi2[i,j], phi3[i,j]], K)
            U[i,j] = dphi2
            V[i,j] = dphi3

    speed = np.sqrt(U**2 + V**2) + epsilon
    U /= speed
    V /= speed
    return phi2, phi3, U, V, speed

#Plots vector field of dphi2 and dphi3 as a streamplot, centered around fp_phi_i
def plot_vector_field(N=30):
    fp_phi2 = np.pi/3
    fp_phi3 = -np.pi/3
    #fig, ax = plt.subplots(figsize=(10, 8))
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8),
                         subplot_kw={"xlabel": r'$\psi_2$',"ylabel": r'$\psi_3$'})
    axes = axes.flatten()
    K_vals = [0,0.15,0.3,0.6,1.0,1.5,2.0,3.0,4.0]

    all_fields = []
    max_speed = 0

    for k in K_vals:
        Phi2, Phi3, U, V, speed = compute_vector_field(k)
        all_fields.append((Phi2, Phi3, U, V, speed))
        max_speed = max(max_speed, np.max(speed))

    for i, (ax, k, (Phi2, Phi3, U, V, speed)) in enumerate(zip(axes, K_vals, all_fields)):
        Phi2, Phi3, U, V, speed = compute_vector_field(k)
        stream = ax.streamplot(Phi2, Phi3, U, V, color=speed,
                               cmap='viridis', density=1.3, arrowsize=1.5,
                               norm=plt.Normalize(vmin=0, vmax=max_speed))
        ax.set_title('$K=$' + str(k))
        ax.set_xlabel(r'$\psi_2$')
        ax.set_ylabel(r'$\psi_3$')
        ax.set_yticks(axis_tick*np.pi)
        ax.set_xticks(axis_tick*np.pi)
        ax.set_yticklabels(axis_label, fontsize=12)
        ax.set_xticklabels(axis_label, fontsize=12)
        ax.set_xlim([fp_phi2-0.2, fp_phi2+0.2])
        ax.set_ylim([fp_phi3-0.2, fp_phi3+0.2])

        if i < 6:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        if i % 3 != 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')


    fig.suptitle(r'Phase flow field around $\psi_j = 0,  \psi_l = \frac{2\pi}{3}$')
    #ax.set_yticks(axis_tick*np.pi)
    #ax.set_xticks(axis_tick*np.pi)
    #ax.set_yticklabels(axis_label, fontsize=12)
    #ax.set_xticklabels(axis_label, fontsize=12)
    #ax.set_aspect('equal')
    # Shared colorbar
    fig.subplots_adjust(right=0.88)  # move the plots left
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(stream.lines, cax=cbar_ax)
    cbar.set_label("Speed")

    plt.show()

    #plt.tight_layout()
    plt.show()
    return stream

# Plots trajectories of each initial condition as arrows
def plot_trajectories(Tmax=40, skip=50):
    N= len(phi2_vals) #Updating the number of angles to sample
    fig, ax = plt.subplots(figsize=(9, 8))
    t_eval = np.linspace(0, Tmax, int(Tmax/0.01)) #Time vector to integrate
    final_states = []
    for phi2_0 in phi2_vals:
        for phi3_0 in phi3_vals:
            sol = solve_ivp(adim_N3_Kuramoto, [0, Tmax], [phi2_0, phi3_0], t_eval=t_eval, method = 'Radau', args = [K,]) #Getting trajectories with Radau (for stiff equations)
            phi2_traj, phi3_traj = sol.y
            #ax.quiver(
            #    phi2_traj[:-skip:skip], phi3_traj[:-skip:skip],
            #    phi2_traj[skip::skip] - phi2_traj[:-skip:skip],
            #    phi3_traj[skip::skip] - phi3_traj[:-skip:skip],
            #    angles='xy', scale_units='xy', scale=1, color='gray', width=0.008, alpha=0.4)
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
    lgd = ax.legend(handles=legend_elements, bbox_to_anchor= (1.35, 0.65), title = "Fixed Points")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_xlabel(r'$\psi_2$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title('Basins of attraction for $K=$' + str(K12))
    ax.set_yticks(axis_tick*np.pi)
    ax.set_xticks(axis_tick*np.pi)
    ax.set_yticklabels(axis_label, fontsize=12)
    ax.set_xticklabels(axis_label, fontsize=12)
    plt.tight_layout()
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, linestyle= ":")

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

    sorted_abs_states = np.array([np.sort(np.abs(row)) for row in states])
    unique_states = np.unique(sorted_abs_states, axis=0)

    labels = []

    for i, (phi_a, phi_b) in enumerate(unique_states, 1):
        # Find all states that match this pair (regardless of order or sign)
        matches = []
        for s in states:
            abs_sorted = np.sort(np.round(np.abs(s), 1))
            if np.allclose(abs_sorted, [phi_a, phi_b]):
                matches.append(s)
        subset = np.array(matches)

        phi_a_str = to_angle(phi_a)
        phi_b_str = to_angle(phi_b)
        label_a = f"±{phi_a_str}" if phi_a != 0 else phi_a_str
        label_b = f"±{phi_b_str}" if phi_b != 0 else phi_b_str

        signs = np.sign(subset)
        sign_products = np.unique(signs[:, 0] * signs[:, 1])

        if len(sign_products) == 1:
            if sign_products[0] == 1:
                label_a = f"±{phi_a_str}"
                label_b = f"±{phi_b_str}"
            elif sign_products[0] == -1:
                label_a = f"±{phi_a_str}"
                label_b = f"∓{phi_b_str}"

        labels.append(fr"{i}: $\psi_i =$ {label_a}, $\psi_j =$ {label_b}")

    return labels

def find_roots(k):
    fixed_points = []

    def system(y):
        return adim_N3_Kuramoto(0, y, k)

    # loop over a subset of grid points as initial guesses
    for i in range(0, phi2.shape[0]):
        for j in range(0, phi2.shape[1]):
            guess = [phi2[i, j], phi3[i, j]]
            sol = root(system, guess)

            if sol.success:
                psi2, psi3 = sol.x
                psi2 = (psi2 + np.pi) % (2*np.pi) - np.pi
                psi3 = (psi3 + np.pi) % (2*np.pi) - np.pi
                fixed = np.round([psi2, psi3], 3)

                if not any(np.allclose(fixed, pt) for pt in fixed_points):
                    fixed_points.append(fixed)

    return np.array(fixed_points)

def classify_fp(fixed_points, k):
    J = sp.Matrix(f_movfr_N3_adim).jacobian((sphi2,sphi3))
    fp_classification = np.full((len(fixed_points),4),-1.0)
    for i,(fp) in enumerate(fixed_points):
        fp_class_temp = np.zeros(4)
        fp_classification[i,0] = fp[0]
        fp_classification[i,1] = fp[1]
        subs = {
            sphi2: fp[0],
            sphi3: fp[1],
            w[0]: 0,
            w[1]: 0,
            w[2]: 0,
            K: k
        }
        J_eval = J.subs(subs)
        trace = J_eval.trace()
        determinant = J_eval.det()
        eigenvalues_s = J_eval.eigenvals()
        discriminant = (trace**2)-(4*determinant)

        eigenvalues = np.array([complex(ev.evalf()) for ev in eigenvalues_s])

        real_parts = np.real(eigenvalues)
        is_stable = np.all(real_parts < 0)

        fp_classification[i, 2] = 1 if is_stable else 0

        if determinant < 0:
            fp_classification[i, 3] = 3  # Saddle
        elif discriminant > 0 and determinant > 0:
            fp_classification[i, 3] = 0  # Node
        elif discriminant < 0 and determinant > 0:
            fp_classification[i, 3] = 1  # Spiral
        elif discriminant == 0 and determinant > 0:
            fp_classification[i, 3] = 2  # Degenerate Node
        elif trace == 0:
            fp_classification[i, 3] = 4  # Line / Center
        else:
            fp_classification[i, 3] = -1  # Undefined/edge case

    return fp_classification

#Plots basins of attraction for trajectories of initial condition sweep
def plot_basins(Tmax=50):

    N= len(phi2_vals) #Updating the number of angles to sample
    fig, ax = plt.subplots(figsize=(9, 8))
    t_eval = np.linspace(0, Tmax, int(Tmax/0.01)) #Time vector to integrate
    final_states = []
    for phi2_0 in phi2_vals:
        for phi3_0 in phi3_vals:
            sol = solve_ivp(adim_N3_Kuramoto, [0, Tmax], [phi2_0, phi3_0], t_eval=t_eval, method = 'Radau', args = [K,]) #Getting trajectories with Radau (for stiff equations)
            phi2_traj, phi3_traj = sol.y
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
    lgd = ax.legend(handles=legend_elements, bbox_to_anchor= (1.35, 0.65), title = "Fixed Points")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_xlabel(r'$\psi_2$')
    ax.set_ylabel(r'$\psi_3$')
    ax.set_title('Basins of attraction for $K=$' + str(K12))
    ax.set_yticks(axis_tick*np.pi)
    ax.set_xticks(axis_tick*np.pi)
    ax.set_yticklabels(axis_label, fontsize=12)
    ax.set_xticklabels(axis_label, fontsize=12)
    plt.tight_layout()
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, linestyle= ":")

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

#plot_trajectories()
#plot_vector_field()
#plot_basins()
plot_nullclines()
