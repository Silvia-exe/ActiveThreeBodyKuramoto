import actkurpy as akpy
import h5py
import plotting as kplt
import numpy as np
import matplotlib.pyplot as plt
#Testing file to work with h5 files.

f = h5py.File('../N1000_run_K_0.00_data.h5')
print(list(f.keys()))
print(list(f["parameters"].keys()))
#print(list(f["dynamics"].keys()))

print("K = ", f["parameters"]["K12"][()])
K12 = f["parameters"]["K12"][()]
print("N = ", f["parameters"]["N"][()])
N = f["parameters"]["N"][()]
print("Lx = ", f["parameters"]["Lx"][()])
Lx = f["parameters"]["Lx"][()]
print("T = ", f["parameters"]["T"][()])
T = f["parameters"]["T"][()]
print("dt = ", f["parameters"]["dt"][()])
dt = f["parameters"]["dt"][()]

act_mat = f["dynamics"]["it0"]["act_mat"][:]
pos_mat = f["dynamics"]["it0"]["pos_mat"][:]
neigh_mat = f["dynamics"]["it0"]["neigh_mat"][:]

print("pos_mat shape", pos_mat[:,:,:].shape)
print("act_mat shape", act_mat[:,:].shape)
print("neigh_mat shape", neigh_mat[:,:,:].shape)
kplt.plot_frames_from_data(act_mat, pos_mat, title = '', Lx = Lx, show = True,  filename = None)

dpos = np.diff(pos_mat, axis = 2)
dtheta = np.diff(act_mat, axis = 1)

dpos -= Lx*np.round(dpos/Lx)
print("shape dtheta:", dtheta.shape)
print("shape dpos:", dpos.shape)
dr = np.sqrt(dpos[:,0,:]*dpos[:,0,:]+dpos[:,1,:]*dpos[:,1,:])
dr[0] = 0
dtheta[0] = 0
print("Mean of theta", np.mean(abs(dtheta)))
print("max(dtheta)=",np.max(abs(dtheta)))
#plt.hist(dtheta.ravel(), density = True)
#plt.xlabel(r'Difference $\Delta \theta = \theta(t+1)-\theta(t)$ [args.]')
#plt.ylabel(r'Density [args.]')
#plt.title(r'Histogram of $\Delta \theta$')
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.plot(np.mean(abs(dtheta), axis = 0))
plt.title(r'Mean abs$(\Delta \theta)$ in time')
plt.xlabel("Time [t]")
plt.ylabel(r'$\Delta \theta$')
print("max(dr)=", np.max(dr))

kplt.plot_phase_coherence_pair_three(act_mat, title = f'Order parameters for K = {K12:.2f}', show = True)
R1_mean, R2_mean, R3_mean = kplt.return_phase_mean(act_mat)
R1_std, R2_std, R3_std = kplt.return_phase_std(act_mat)

print("<R1> = ", R1_mean)
print("<R2> = ", R2_mean)
print("<R3> = ", R3_mean)

print(r'$\sigma$ (R1) =', R1_std)
print(r'$\sigma$ (R2) =', R2_std)
print(r'$\sigma$ (R3) =', R3_std)

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
