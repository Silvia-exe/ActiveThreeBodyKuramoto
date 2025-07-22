from actkurpy import ActiveKuramotoAdim
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
import plotting as kplt
import numpy as np
import h5py
import random
from concurrent.futures import ProcessPoolExecutor
import os
from scipy.spatial import ConvexHull

def simple_run():
    for k in range(len(K12_v)):
        K12 = K12_v[k]
        print("Simple run, running for K = ", K12)
        run = ActiveKuramotoAdim(N,K12,Lx,gamma,mov_coef,dt,T)
        act_mat, pos_mat, neigh_mat = run.run()
        filename = f"K_{K12:.2f}"
        #kplt.animate_active_oscillators(act_mat, pos_mat, Lx = Lx, title = f'Evolution for K ={K12:.2f}', show = True )
        kplt.plot_phase_coherence_pair_three(act_mat, title = f'Order parameters for K = {K12:.2f}', show = False, filename = 'order_par_'+filename)
        kplt.plot_frames_from_data(act_mat, pos_mat,filename = 'frames_'+filename, Lx = Lx, title = f'Evolution for K ={K12:.2f}', show = False)
        run.save_data('N1000_run_'+filename+'_data.h5')

def run_plot_all():
    _, Kax = plt.subplots()
    normalize = colors.Normalize(vmin=np.min(K12_v), vmax=np.max(K12_v))
    cmap = cm.magma

    for k in range(len(K12_v)):
        K12 = K12_v[k]
        filename = f"K_{K12:.2f}"
        color = cmap(normalize(K12))
        run = ActiveKuramotoAdim(N,K12,Lx,gamma,mov_coef,dt,T)
        print(K12)
        act_mat, pos_mat,  = run.run()
        #run.save_data('model_K_'+filename+'_data.h5')
        #kplt.animate_active_oscillators(act_mat, pos_mat, filename= 'animation_'+filename, Lx = Lx, title = f'Evolution for K ={K12:.2f}', show = False )
        kplt.plot_frames_from_data(act_mat, pos_mat,filename = 'frames_'+filename, Lx = Lx, title = f'Evolution for K ={K12:.2f}', show = False)
        kplt.plot_phase_coherence_pair_three(act_mat, title = f'Order parameters for K = {K12:.2f}', filename = 'order_par_'+filename, show = False)
        Kax.plot([kplt.phase_coherence(vec) for vec in act_mat.T], '.', label = f"K = {K12:.2f}", color = color)
        plt.close()

    Kax.set_title(r"Order Parameter $R_1(t)$ for different $K = \frac{K_1}{K_2}$")
    sm = cm.ScalarMappable(cmap=cmap, norm=normalize)
    cbar = plt.colorbar(sm, ax=Kax)
    cbar.set_label(r"$K=\frac{K_1}{K_2}$", rotation = 90,  labelpad=15)
    #cbar.set_label(r"$K = \frac{K_1}{K_2}$", rotation = 90,  labelpad=15)
    plt.savefig('R1_of_K'+filename+ ".png")
    #plt.show()

def run_average_R():
    _, ax = plt.subplots()
    R1_K_mean = np.zeros((len(K12_v), M))
    R2_K_mean = np.zeros((len(K12_v), M))
    R3_K_mean = np.zeros((len(K12_v), M))

    R1_K = np.zeros(len(K12_v))
    R2_K = np.zeros(len(K12_v))
    R3_K = np.zeros(len(K12_v))

    for k in range(len(K12_v)):
        print("Here")
        K12 = K12_v[k]
        filename = f"model_K_{K12:.2f}_data.h5"
        print(K12)
        for l in range(M):
            print(l)
            run = ActiveKuramotoAdim(N,K12,Lx,gamma, mov_coef,dt,T)
            act_mat, pos_mat,  = run.run()
            run.reset()
            R1_K_mean[k, l], R2_K_mean[k,l], R3_K_mean[k,l] = kplt.return_phase_mean(act_mat)
            if l%10 == 0:
                with h5py.File(filename, "a") as dyn:
                    dyn.create_group("dynamics/it"+str(l))
                    dyn.create_dataset("dynamics/it"+str(l)+"/act_mat", data = act_mat, compression='gzip', dtype = np.single, chunks = True, compression_opts = 6)
                    dyn.create_dataset("dynamics/it"+str(l)+"/pos_mat", data = pos_mat, compression='gzip', dtype = np.single, chunks = True, compression_opts = 6)

        R1_K[k] = np.mean(R1_K_mean[k])
        R2_K[k] = np.mean(R2_K_mean[k])
        R3_K[k] = np.mean(R3_K_mean[k])

    filename_K = "order_par_means_K.h5"
    with h5py.File(filename_K, "w") as h5f:
        h5f.create_group("parameters")

        para_list = []
        para_list += [("/parameters/N", N)]
        para_list += [("/parameters/T", T)]
        para_list += [("/parameters/dt", dt)]
        para_list += [("/parameters/Lx", Lx)]
        para_list += [("/parameters/gamma", gamma)]

        for element in para_list:
            h5f.create_dataset(element[0],data=element[1])

        h5f.create_group("K_val")
        h5f.create_group("R1")
        h5f.create_group("R2")
        h5f.create_group("R3")

        h5f.create_dataset("K_val/K_vals", data = K12_v, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)
        h5f.create_dataset("R1/R1_means", data = R1_K_mean, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)
        h5f.create_dataset("R2/R2_means", data = R2_K_mean, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)
        h5f.create_dataset("R3/R3_means", data = R3_K_mean, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)

    plt.plot(K12_v, R1_K, label = "$R_1(K)$" )
    plt.plot(K12_v, R2_K, label = "$R_2(K)$" )
    plt.plot(K12_v, R3_K, label = "$R_3(K)$" )
    plt.xlabel(r"$K = \frac{K_1}{K_2}$")
    plt.ylabel(r"Order Parameters $<R_i>$")
    plt.legend(title = "Order parameter")
    plt.show()

def single_run(args):
    os.system("cls")
    k, l, K12, N, Lx, gamma, dt, T = args
    print(K12)
    print(l)
    run = ActiveKuramotoAdim(N, K12, Lx, gamma, mov_coef, dt, T)
    act_mat, pos_mat,  = run.run()
    R1, R2, R3 = kplt.return_phase_mean(act_mat)
    '''if l%10 == 0:
        filename = f"model_K_{K12:.2f}_data_N1k.h5"
        with h5py.File(filename, "a") as dyn:
            dyn.create_group("dynamics/it"+str(l))
            dyn.create_dataset("dynamics/it"+str(l)+"/act_mat", data = act_mat, compression='gzip', dtype = np.single, chunks = True, compression_opts = 6)
            dyn.create_dataset("dynamics/it"+str(l)+"/pos_mat", data = pos_mat, compression='gzip', dtype = np.single, chunks = True, compression_opts = 6)'''
    return (k, l, R1, R2, R3)

def run_average_R_parallel():
    R1_K_mean = np.zeros((len(K12_v), M), dtype = np.single)
    R2_K_mean = np.zeros((len(K12_v), M), dtype = np.single)
    R3_K_mean = np.zeros((len(K12_v), M), dtype = np.single)

    R1_K = np.zeros(len(K12_v), dtype = np.single)
    R2_K = np.zeros(len(K12_v), dtype = np.single)
    R3_K = np.zeros(len(K12_v), dtype = np.single)

    # Collect all simulation tasks
    all_jobs = []
    for k, K12 in enumerate(K12_v):
        for l in range(M):
            all_jobs.append((k, l, K12, N, Lx, gamma, dt, T))

    # Run in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()-1 or 1) as executor:
        for k, l, R1, R2, R3 in executor.map(single_run, all_jobs, chunksize = 10):
            R1_K_mean[k, l] = R1
            R2_K_mean[k, l] = R2
            R3_K_mean[k, l] = R3

    # Average across trials
    for k in range(len(K12_v)):
        R1_K[k] = np.mean(R1_K_mean[k])
        R2_K[k] = np.mean(R2_K_mean[k])
        R3_K[k] = np.mean(R3_K_mean[k])

    # Save summary results
    filename = "order_par_means_K_rho4_N100"
    with h5py.File(filename+".h5", "w") as h5f:
        h5f.create_group("parameters")
        para_list = []
        para_list += [("/parameters/N", N)]
        para_list += [("/parameters/T", T)]
        para_list += [("/parameters/dt", dt)]
        para_list += [("/parameters/Lx", Lx)]
        para_list += [("/parameters/gamma", gamma)]

        for element in para_list:
            h5f.create_dataset(element[0],data=element[1])

        h5f.create_group("K_val")
        h5f.create_group("R1_means")
        h5f.create_group("R2_means")
        h5f.create_group("R3_means")

        h5f.create_dataset("K_val/K_vals", data=K12_v, compression="gzip", dtype=np.single, chunks=True, compression_opts=6)
        h5f.create_dataset("R1_means/R1", data=R1_K_mean, compression="gzip", dtype=np.single, chunks=True, compression_opts=6)
        h5f.create_dataset("R2_means/R2", data=R2_K_mean, compression="gzip", dtype=np.single, chunks=True, compression_opts=6)
        h5f.create_dataset("R3_means/R3", data=R3_K_mean, compression="gzip", dtype=np.single, chunks=True, compression_opts=6)

    # Plot
    plt.plot(K12_v, R1_K, label="$R_1(K)$")
    plt.plot(K12_v, R2_K, label="$R_2(K)$")
    plt.plot(K12_v, R3_K, label="$R_3(K)$")
    plt.xlabel(r"$K = \frac{K_1}{K_2}$")
    plt.ylabel(r"Order Parameters $<R_i>$")
    plt.legend(title="Order parameter")
    plt.show()
    plt.savefig(filename+".png")

def find_cluster(particle_id, neig_mat, t):
    visited = set()
    to_visit = [particle_id]

    while to_visit:
        current = to_visit.pop()
        if current not in visited:
            visited.add(current)
            neigh = neig_mat[current, :, t]
            for n in neigh:
                if n not in visited:
                    to_visit.append(n)

    return list(visited)

def find_all_clusters(neigh_mat, t):
    """
    Find all clusters (connected components) at time t
    based on directed neighbor relations.

    Returns a list of lists, where each sublist is a cluster (list of particle indices).
    """
    N = neigh_mat.shape[0]
    visited = set()
    clusters = []

    def dfs(start, cluster):
        to_visit = [start]
        while to_visit:
            current = to_visit.pop()
            if current not in cluster:
                cluster.add(current)
                neighbors = neigh_mat[current, :, t]
                for n in neighbors:
                    if n not in cluster:
                        to_visit.append(n)

    for i in range(N):
        if i not in visited:
            cluster = set()
            dfs(i, cluster)
            visited.update(cluster)
            clusters.append(list(cluster))

    return clusters

def get_distinct_colors(n):
    random.seed(42)  # for reproducibility
    colors = []
    for _ in range(n):
        r, g, b = [random.random() for _ in range(3)]
        colors.append((r, g, b))
    return colors

def plot_clusters(pos_mat, clusters, t):
    cluster_colors = get_distinct_colors(len(clusters))
    plt.figure(figsize=(6, 6))

    for i, cluster in enumerate(clusters):
        xs = pos_mat[cluster, 0, t]
        ys = pos_mat[cluster, 1, t]
        plt.scatter(xs, ys, color=cluster_colors[i], s=30)

        # Draw convex hull if 3 or more points
        if len(cluster) >= 3:
            points = np.column_stack((xs, ys))
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], color=cluster_colors[i], linewidth=2)

    plt.title(f"Clusters at time {t}")
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_single_cluster(pos_mat, cluster, t):
    """
    positions: np.array of shape (N, 2, T)
    cluster_indices: list of indices of particles in the cluster
    t_index: time index to plot
    Lx: box size (for plot limits)
    """
    # Extract all particle positions at time t
    all_pos = pos_mat[:, :, t]
    cluster_pos = all_pos[cluster]

    plt.figure(figsize=(5,5))
    plt.scatter(all_pos[:, 0], all_pos[:, 1], color='gray', alpha=0.4, label='All particles')
    plt.scatter(cluster_pos[:, 0], cluster_pos[:, 1], color='red', label='Cluster', s=60)

    plt.xlim(0, Lx)
    plt.ylim(0, Lx)
    plt.gca().set_aspect('equal')
    plt.title(f"Cluster at time {t}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N = 100
    M = 50
    K1 = 1.0
    K2 = 3.0
    #K12_v = [0.0,4.0]
    K12 = 4.0
    #K12_v = np.linspace(0, 4.0, 3)
    Lx = 5
    v = 1.0
    mov_coef = 1.0
    gamma = 10
    dt = 0.0005
    T = 150

    #simple_run()
    #init_cond = ActiveKuramotoAdim(N,K12,Lx,gamma,mov_coef,dt,T)
    #init_cond.save_data("initial_conditions_N100.h5")

    run = ActiveKuramotoAdim(N,K12,Lx,gamma,mov_coef,dt,T)
    kplt.plot_frames_from_data(run.act_mat, run.pos_mat, title = '', Lx = Lx, show = True,  filename = None)
    run.initialize_from_h5("../initial_conditions_N100.h5")
    kplt.plot_frames_from_data(run.act_mat, run.pos_mat, title = '', Lx = Lx, show = True,  filename = None)
    print("Running for ", run.K12)

    act_mat, pos_mat, neigh_mat = run.run()

    filename = f"dt0-0005_test_K_{K12:.2f}"
    #kplt.animate_active_oscillators(act_mat, pos_mat, Lx = Lx, title = f'Evolution for K ={K12:.2f}', show = True, filename = filename )
    kplt.plot_phase_coherence_pair_three(act_mat, title = f'Order parameters for K = {K12:.2f}', show = True, filename = filename )
    run.save_data(filename+'_data.h5')

    #print(run.T)

    #cluster = find_cluster(62, neigh_mat, int(T/dt)-1)
    #plot_single_cluster(pos_mat,cluster,int(T/dt)-1)

    #clusters = find_all_clusters(neigh_mat,int(T/dt)-1)
    #plot_clusters(pos_mat,clusters,int(T/dt)-1)
    #run_plot_all()
    #run_average_R()
    #run_average_R_parallel()
