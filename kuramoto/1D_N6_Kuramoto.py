import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import plotting as kplt
from scipy.integrate import odeint
from kuramoto import Kuramoto

# === Parameters ===
N = 6
K2 = 1.0
K3 = 10.0
nat_freq = np.zeros(N)

# === Interaction Functions ===
#Interactions with immediate neighbors
def pairwise_interactions(angles):
    interactions = np.zeros(N)
    for i in range(N):
        interactions[i] = np.sin(angles[(i-1)%N]-angles[i]) + np.sin(angles[(i+1)%N]-angles[i])
    return interactions

def threewise_interactions(angles):
    interactions = np.zeros(N)
    for i in range(N):
        interactions[i] = (
            np.sin(2*angles[(i-1)%N] - angles[(i-2)%N] - angles[i]) + np.sin(2*angles[(i-2)%N] - angles[(i-1)%N] - angles[i]) + \
            np.sin(2*angles[(i-1)%N] - angles[(i+1)%N] - angles[i]) + np.sin(2*angles[(i+1)%N] - angles[(i-1)%N] - angles[i]) + \
            np.sin(2*angles[(i+1)%N] - angles[(i+2)%N] - angles[i]) + np.sin(2*angles[(i+1)%N] - angles[(i+2)%N] - angles[i])
        )
    return interactions

# === Dynamical Derivative ===
def derivative(angles,t):
    return nat_freq + 1/2 * K1 * pairwise_interactions(angles) + 1/6 * K2 * threewise_interactions(angles)

def integrate(angles,t):
    timeseries = odeint(derivative, angles, t)
    return timeseries.T

# === Wrapper with theta_0 = 0 fixed ===
def wrapped_derivative(theta_rest):
    theta_full = np.concatenate([[0], theta_rest])
    return derivative(theta_full)[1:]  # ignore theta_0 derivative (fixed)

# === Fixed Point Finder ===
def find_fixed_points(n_trials=1000, tol=1e-6):
    fixed_points = []
    for _ in range(n_trials):
        guess = np.random.uniform(-np.pi, np.pi, N-1)
        sol = root(wrapped_derivative, guess, method='hybr')
        if sol.success and np.linalg.norm(wrapped_derivative(sol.x)) < tol:
            full = np.concatenate([[0], sol.x])
            normed = normalize(full)
            if not any(np.allclose(normed, fp, atol=1e-2) for fp in fixed_points):
                fixed_points.append(normed)
    print(fixed_points)
    return fixed_points

# === Normalization and Order Parameter ===
def normalize(theta):
    return np.mod(theta - theta[0], 2*np.pi)

def order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

# === Cluster + Analyze ===
def analyze_fixed_points(fixed_points):
    fps_normalized = np.array(fixed_points)

    clustering = AgglomerativeClustering(distance_threshold=0.5, n_clusters=None)
    labels = clustering.fit_predict(fps_normalized)
    n_clusters = len(set(labels))
    print(f"🔍 Found {n_clusters} clusters")

    # PCA
    pca = PCA(n_components=2)
    fps_2d = pca.fit_transform(fps_normalized)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(fps_2d[:, 0], fps_2d[:, 1], c=labels, cmap='tab10', s=60)
    plt.title("Fixed Points Clustering (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.colorbar(scatter, label='Cluster ID')
    plt.show()

    # Dendrogram
    Z = linkage(fps_normalized, method='ward')
    plt.figure(figsize=(12, 4))
    dendrogram(Z)
    plt.title("Dendrogram of Fixed Points")
    plt.xlabel("Fixed Point Index")
    plt.ylabel("Distance")
    plt.show()

    # Cluster summaries
    for i in range(n_clusters):
        print(f"\n🌀 Cluster {i} Summary:")
        cluster_indices = np.where(labels == i)[0]
        print(f" - Contains {len(cluster_indices)} fixed points")

        fp_example = fps_normalized[cluster_indices[0]]
        R = order_parameter(fp_example)
        print(f" - Order parameter R ≈ {R:.3f}")
        plot_on_circle(fp_example, title=f"Cluster {i}: Example Fixed Point")

def epsilon_i():
    return 0.005*2*np.pi*np.random.uniform(-1,1)

N = 6
T = 10
dt = 0.01
t = np.linspace(0,T,int(T/dt))
nat_freq = np.zeros(6)
init_angle= 2*np.pi/3

K1 = 1
K2 = 2 #[0,1,2,5,10,50] #1000

#init_angles = np.array([2*np.pi*np.random.uniform(-1,1) for _ in range(N)])

#Theta1 will be used as reference moving frame
init_angles = [0, 2*np.pi/6+epsilon_i(), 4*np.pi/6+epsilon_i(), 6*np.pi/6+epsilon_i(), 8*np.pi/6+epsilon_i(), 10*np.pi/6+epsilon_i()] #np.random.uniform(-np.pi,np.pi,N)
print("Initial angles: ", init_angles)

act_mat = integrate(init_angles,t)
print("Final angles:", act_mat.T[-1])

kplt.animate_oscillators(act_mat)
