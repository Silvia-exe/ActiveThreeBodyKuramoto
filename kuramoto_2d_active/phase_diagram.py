import actkurpy as akpy
import h5py
import plotting as kplt
import numpy as np
import matplotlib.pyplot as plt


f04 = h5py.File('./K_sweeps_Ri/order_par_means_K_rho0-4_N100.h5')
f4 = h5py.File('./K_sweeps_Ri/order_par_means_K_rho4_N100.h5')
f40 = h5py.File('./K_sweeps_Ri/order_par_means_K_rho40_N100.h5')

R1_means_04 = f04["R1_means"]["R1"][:]
R2_means_04 = f04["R2_means"]["R2"][:]
R3_means_04 = f04["R3_means"]["R3"][:]

R1_means_4 = f4["R1_means"]["R1"][:]
R2_means_4 = f4["R2_means"]["R2"][:]
R3_means_4 = f4["R3_means"]["R3"][:]

R1_means_40 = f40["R1_means"]["R1"][:]
R2_means_40 = f40["R2_means"]["R2"][:]
R3_means_40 = f40["R3_means"]["R3"][:]

K_vals = f04["K_val"]["K_vals"][:]

R1_K_04 = np.mean(R1_means_04, axis = 1)
R2_K_04 = np.mean(R2_means_04, axis = 1)
R3_K_04 = np.mean(R3_means_04, axis = 1)

R1_K_4 = np.mean(R1_means_4, axis = 1)
R2_K_4 = np.mean(R2_means_4, axis = 1)
R3_K_4 = np.mean(R3_means_4, axis = 1)

R1_K_40 = np.mean(R1_means_40, axis = 1)
R2_K_40 = np.mean(R2_means_40, axis =1)
R3_K_40 = np.mean(R3_means_40, axis = 1)

# Plot
'''plt.plot(K_vals, R1_K, label="$R_1(K)$")
plt.plot(K_vals, R2_K, label="$R_2(K)$")
plt.plot(K_vals, R3_K, label="$R_3(K)$")
plt.xlabel(r"$K = \frac{K_1}{K_2}$")
plt.ylabel(r"Order Parameters $<R_i>$")
plt.legend(title="Order parameter")
plt.show()'''

critK_04 = np.where(R1_K_04 > R2_K_04)[0][0]
critK_4 = np.where(R1_K_4 > R2_K_4)[0][0]
critK_40 = np.where(R1_K_40 > R2_K_40)[0][0]

plt.plot([K_vals[critK_04],K_vals[critK_4],K_vals[critK_40]],[0.4,4,40])

plt.title("N = 100, critical K values for transition.")
plt.xlabel("K value")
plt.xlim(0.5,2.0)
plt.ylabel(r"$\rho$ value")
plt.show()
