from actkurpy import ActiveKuramotoAdim
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
import plotting as kplt
import numpy as np

N = 100
M = 1
K1 = 1.0
K2 = 3.0
K12_v = np.linspace(0, 10, 20)
Lx = 5
v = 1.0
gamma = 10
dt = 0.01
T = 150

_, Kax = plt.subplots()
normalize = colors.Normalize(vmin=np.min(K12_v), vmax=np.max(K12_v))
cmap = cm.magma

for k in range(len(K12_v)):
    K12 = K12_v[k]
    filename = f"K_{K12:.2f}"
    color = cmap(normalize(K12))
    run = ActiveKuramotoAdim(N,K12,Lx,gamma,dt,T)
    print(K12)
    for l in range(M):
        act_mat, pos_mat = run.run()
    kplt.animate_active_oscillators(act_mat, pos_mat, filename= 'animation_'+filename, title = f'Evolution for K ={K12:.2f}', show = False )
    kplt.plot_frames_from_data(act_mat, pos_mat,filename = 'frames_'+filename, title = f'Evolution for K ={K12:.2f}', show = False)
    kplt.plot_phase_coherence_pair_three(act_mat, title = f'Order parameters for K = {K12:.2f}', filename = 'order_par_'+filename, show = False)
    Kax.plot([kplt.phase_coherence(vec) for vec in act_mat.T], '.', label = f"K = {K12:.2f}", color = color)
    plt.close()

Kax.set_title(r"Order Parameter $R_1(t)$ for different $K = \frac{K_1}{K_2}$")
sm = cm.ScalarMappable(cmap=cmap, norm=normalize)
cbar = plt.colorbar(sm, ax=Kax)
cbar.set_label(r"$K=\frac{K_1}{K_2}$", rotation = 90,  labelpad=15)
#cbar.set_label(r"$K = \frac{K_1}{K_2}$", rotation = 90,  labelpad=15)
plt.savefig('R1_of_K'+filename)
plt.show()
