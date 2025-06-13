import actkurpy as akpy
import h5py
import plotting as kplt

N = 100
K12 = 3
Lx = 5.0
gamma = 10.0
dt = 0.1
T = 50

#test = akpy.ActiveKuramotoAdim(N,K12,Lx,gamma,dt,T)
#test.run()
#test.save_data("testing_h5.h5")

f = h5py.File('testing_h5.h5')
print(list(f["parameters"].keys()))
print(f["parameters"]["N"][()])
print(f["parameters"]["Lx"][()])
print(f["parameters"]["T"][()])
print(f["parameters"]["dt"][()])

act_mat = f["dynamics"]["act_mat"]
pos_mat = f["dynamics"]["pos_mat"]

kplt.animate_active_oscillators(act_mat = act_mat, pos_mat = pos_mat, show = True, title= " ", dt = f["parameters"]["dt"][()])
