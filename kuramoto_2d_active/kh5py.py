import actkurpy as akpy
import h5py
import plotting as kplt
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from collections import defaultdict
#Testing file to work with h5 files.

def hash_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

def plot_order_parameters_scaled(act_mat_list, dt_list, K_val, labels=None):
    assert len(act_mat_list) == len(dt_list), "Mismatch between act_mats and dt_list"

    if labels is None:
        labels = [f'dt={dt}' for dt in dt_list]

    # Step 1: call your original plotting function, suppress output
    axes_raw = [
        kplt.plot_phase_coherence_pair_three(act_mat, title='', show=False)
        for act_mat in act_mat_list
    ]

    # Step 2: prepare new figure with 3 subplots for R1, R2, R3
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Step 3: helper to extract line from each axis and replot with proper time axis
    def copy_and_rescale_line(line_idx, target_ax, ylabel):
        for i, (ax_raw, act_mat, dt) in enumerate(zip(axes_raw, act_mat_list, dt_list)):
            line = ax_raw.get_lines()[line_idx]
            ydata = line.get_ydata()
            t = np.arange(len(ydata)) * 2*dt
            target_ax.plot(t, ydata, label=labels[i])
        target_ax.set_ylabel(ylabel, fontsize=14)
        target_ax.legend()

    # Plot R1, R2, R3
    copy_and_rescale_line(0, axes[0], r'$R_1(\theta)$')
    copy_and_rescale_line(1, axes[1], r'$R_2(\theta)$')
    copy_and_rescale_line(2, axes[2], r'$R_3(\theta)$')

    # Final formatting
    axes[2].set_xlabel('Time [t]', fontsize=14)
    fig.suptitle(f'Order parameters for K = {K_val:.2f}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def dt_test(file_list):
    act_mat_list = [f["dynamics"]["it0"]["act_mat"][:] for f in file_list]
    dt_list = [f["parameters"]["dt"][()] for f in file_list]
    K_val = np.array([f["parameters"]["K12"][()] for f in file_list])
    K_val = np.unique(K_val)
    assert len(K_val) == 1, "Running dt test for different values of K?"
    hashes = [hash_array(act_mat[:, 0]) for act_mat in act_mat_list]
    # Group indices by hash
    hash_groups = defaultdict(list)
    for i, h in enumerate(hashes):
        hash_groups[h].append(i)

    # Print the result
    print("Initial condition groups:")
    for group_id, (h, indices) in enumerate(hash_groups.items()):
        print(f"Group {group_id + 1} (hash={h[:8]}): simulations {indices}")

    plot_order_parameters_scaled(act_mat_list,dt_list, K_val[0])

def print_param(file):
    h = file
    print("K = ", h["parameters"]["K12"][()])
    print("N = ", h["parameters"]["N"][()])
    print("Lx = ", h["parameters"]["Lx"][()])
    print("T = ", h["parameters"]["T"][()])
    print("dt = ", h["parameters"]["dt"][()])

def dt_plots(file):
    act_mat = file["dynamics"]["it0"]["act_mat"][:]
    pos_mat = file["dynamics"]["it0"]["pos_mat"][:]
    K12 = file["parameters"]["K12"][()]
    dt = file["parameters"]["dt"][()]
    dpos = np.diff(pos_mat, axis = 2)
    dpos -= Lx*np.round(dpos/Lx)
    dtheta = np.diff(act_mat, axis = 1)
    dr = np.sqrt(dpos[:,0,:]**2+dpos[:,1,:]**2)
    dr = np.delete(dr, 0)
    dtheta = np.delete(dtheta, 0)
    print(r"Mean of \theta: ", np.mean(abs(dtheta/2)))
    print("Mean of dr: ", np.mean(abs(dr/2)))
    print(r"max(d\theta)=",np.max(abs(dtheta/2)))
    print("max(dr)=", np.max(dr/2))

    plt.hist(dr.ravel()/2, density = True)
    plt.xlabel(r'Difference $\Delta r = r(t+1)-r(t)$ [args.]')
    plt.ylabel(r'Density [args.]')
    plt.title(rf'Histogram of $\Delta r$ for dt = {dt} and K = {K12}')
    plt.savefig(f'dr_hist_K_{K12}_dt_{dt}.png')
    plt.show()
    plt.clf()

    plt.hist(dtheta.ravel()/2, density = True)
    plt.xlabel(r'Difference $\Delta \theta = \theta(t+1)-\theta(t)$ [args.]')
    plt.ylabel(r'Density [args.]')
    plt.title(rf'Histogram of $\Delta \theta$ for dt = {dt} and K = {K12}')
    plt.savefig(f'dtheta_hist_K_{K12}_dt_{dt}.png')
    plt.show()
    plt.clf()

    plt.plot(np.mean(abs(dr/2), axis = 0))
    plt.xlabel("Time [t]")
    plt.ylabel(r'$\Delta r$')
    plt.title(rf'Mean abs$(\Delta r)$ in time for dt = {dt} and K = {K12}')
    plt.savefig(f'dr_mean_plot_K_{K12}_dt_{dt}.png')
    plt.show()
    plt.clf()

    plt.plot(np.mean(abs(dtheta/2), axis = 0))
    plt.xlabel("Time [t]")
    plt.ylabel(r'$\Delta \theta$')
    plt.title(rf'Mean abs$(\Delta \theta)$ in time for dt = {dt} and K = {K12}')
    plt.savefig(f'dtheta_mean_plot_K_{K12}_dt_{dt}.png')
    plt.show()
    plt.clf()

f = h5py.File('./run_dt_0.1_K_4.0_N_100_data.h5')
g = h5py.File('./run_dt_0.05_K_4.0_N_100_data.h5')
h = h5py.File('./run_dt_0.02_K_4.0_N_100_data.h5')
i = h5py.File('./run_dt_0.01_K_4.0_N_100_data.h5')
j = h5py.File('./run_dt_0.005_K_4.0_N_100_data.h5')
k = h5py.File('./run_dt_0.0005_K_4.0_N_100_data.h5')

file_list = [f,g,h,i,j,k]

dt_test(file_list)

#R1_mean, R2_mean, R3_mean = kplt.return_phase_mean(act_mat)
#R1_std, R2_std, R3_std = kplt.return_phase_std(act_mat)

# print("<R_1> = ", R1_mean)
# print("<R_2> = ", R2_mean)
# print("<R_3> = ", R3_mean)
#
# print(r'\sigma (R_1) =', R1_std)
# print(r'\sigma (R_2) =', R2_std)
# print(r'\sigma (R_3) =', R3_std)

'''print(f["K_val"]["K_vals"][:])
print(f["R1"]["R1_means"][:])

R1_mean = 0
R2_mean = 0
R3_mean = 0

i = 0

for it in f["dynamics"]:
    i += 1
    act_mat = f["dynamics"][it]["act_mat"][:]
    pos_mat = f["dynamics"][it]["pos_mat"][:]
    kplt.animate_active_oscillators(act_mat = act_mat, pos_mat = pos_mat, show = True, title= " ", dt = f["parameters"]["dt"][()])
    R1_mean_t, R2_mean_t, R3_mean_t = kplt.return_phase_mean(act_mat)
    print("_________________")
    print(R1_mean_t, R2_mean_t, R3_mean_t)
    R1_mean += R1_mean_t
    R2_mean += R2_mean_t
    R3_mean += R3_mean_t
    kplt.plot_phase_coherence_pair_three(act_mat, title = "")

print(R1_mean/i)
print(R2_mean/i)
print(R3_mean/i)'''
