import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
import plotting as kplt
import numpy as np
import h5py
import math
from scipy.spatial import cKDTree
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe'

'''Active 2D Kuramoto model class
Models a system of coupled active oscillators. The movement direction is given by theta.
The oscillators are coupled by pairwise and threewise interactions.
N = Number of particles (def = 100)
K1 = Coupling parameter for pairwise interactions
K2 = Coupling parameter for trio interactions
v = Particle self-propelling velocity
gamma = Noise strength WIP
dt = Timestep
T = Total simulation time

Returns act_mat and pos_mat, arrays of oscillator phase and position in time.'''

#Adimensionalized Kuramoto equation implementation.
class ActiveKuramotoAdim:
    def __init__(self, N, K12, Lx, gamma, mov_coef, dt, T):
        self.N = N
        self.Lx = Lx #In units of ell = 2v/K2
        self.K12 = K12
        self.gamma = gamma
        self.mov_coef = mov_coef
        self.dt = dt
        self.T = T
        self.t = np.linspace(0, self.T, int(self.T / self.dt))
        self.r, self.theta = self.initialize()
        self.pos_mat = np.zeros((self.N, 2, len(self.t) + 1))
        self.pos_mat[:, :, 0] = self.r
        self.act_mat = np.zeros((self.N, len(self.t)+1))
        self.act_mat[:,0] = self.theta
        self.neigh = self.find_neighbors()
        self.neigh_mat = np.zeros((self.N, 2, len(self.t) + 1), dtype=int)
        self.neigh_mat[:, :, 0] = self.neigh

    def initialize(self):
        r0 = (np.random.rand(self.N, 2) * self.Lx)%self.Lx
        theta0 = np.random.uniform(-np.pi, np.pi, self.N)
        return r0, theta0

    def initialize_from_h5(self, h5_init_file):
         with h5py.File(h5_init_file, 'r') as f:
            act_mat_f = f["dynamics"]["it0"]["act_mat"][:]
            pos_mat_f = f["dynamics"]["it0"]["pos_mat"][:]
            neigh_mat_f = f["dynamics"]["it0"]["neigh_mat"][:]

            self.theta = act_mat_f[:, 0]
            self.r = pos_mat_f[:, :, 0] % self.Lx
            self.neigh = neigh_mat_f[:, :, 0].astype(int)

            self.t = np.linspace(0, self.T, int(self.T / self.dt))
            self.pos_mat = np.zeros((self.N, 2, len(self.t) + 1))
            self.act_mat = np.zeros((self.N, len(self.t)+1))
            self.neigh_mat = np.zeros((self.N, 2, len(self.t) + 1), dtype=int)

            self.act_mat[:, 0] = self.theta
            self.pos_mat[:, :, 0] = self.r
            self.neigh_mat[:, :, 0] = self.neigh

    def find_neighbors(self):
        self.r = self.r % self.Lx
        tree = cKDTree(self.r, boxsize=self.Lx)  # Periodic boundary
        _, indices = tree.query(self.r, k=3)  # self + 2 nearest
        return indices[:, 1:]  # drop self

    def move(self, r, th):
        dx = self.dt * np.cos(th) * self.mov_coef
        dy = self.dt * np.sin(th) * self.mov_coef
        return (r + np.stack((dx, dy), axis=1)) % self.Lx

    def move_noise(self, r, th):
        new_r = np.copy(r)
        for i in range(self.N):
            nx, ny = self.box_muller_noise()
            dx = self.mov_coef*self.dt*np.cos(th[i]) + np.sqrt(2*(1/self.gamma)*self.dt) * nx
            dy = self.mov_coef*self.dt*np.sin(th[i]) + np.sqrt(2*(1/self.gamma)*self.dt) * ny
            #dx = v*dt + 1/gamma * nx
            #dy = v*dt + 1/gamma * ny
            new_r[i] += [dx, dy]
        return new_r%self.Lx

    def kuramoto_neigh(self, th, neigh):
        def dth(th):
            th_l = th[neigh[:, 0].astype(int)]
            th_r = th[neigh[:, 1].astype(int)]
            dth = self.K12 * (np.sin(th_l - th) + np.sin(th_r - th))
            dth += np.sin(2 * th_r - th_l - th) + np.sin(2 * th_l - th_r - th)
            return dth
        k1 = self.dt * dth(th)
        k2 = self.dt * dth(th + 0.5 * k1)
        k3 = self.dt * dth(th + 0.5 * k2)
        k4 = self.dt * dth(th + k3)
        return th + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def run(self):
        for i, _ in enumerate(self.t):
            self.theta = self.kuramoto_neigh(self.theta, self.neigh)
            self.r = self.move(self.r, self.theta)
            self.neigh = self.find_neighbors()
            self.pos_mat[:,:,i+1] = self.r
            self.act_mat[:,i+1] = self.theta
            self.neigh_mat[:,:,i+1] = self.neigh
        return self.act_mat, self.pos_mat, self.neigh_mat

    def run_noise(self):
        for i, _ in enumerate(self.t):
            self.theta = self.kuramoto_neigh(self.theta, self.neigh)
            self.r = self.move_noise(self.r, self.theta)
            self.neigh = self.find_neighbors()
            self.pos_mat[:,:,i+1] = self.r
            self.act_mat[:,i+1] = self.theta
            self.neigh_mat[:,:,i + 1] = self.neigh
        return self.act_mat, self.pos_mat, self.neigh_mat

    def reset(self):
        self.r, self.theta = self.initialize()
        self.pos_mat = np.zeros((self.N, 2, len(self.t) + 1))
        self.pos_mat[:, :, 0] = self.r
        self.act_mat = np.zeros((self.N, len(self.t)+1))
        self.neigh = self.find_neighbors()
        self.neigh_mat = np.zeros((self.N, 2, len(self.t) + 1), dtype=int)
        self.neigh_mat[:, :, 0] = self.neigh

    def save_data(self, filename):
        with h5py.File("../"+filename+'.h5', "w") as h5f:
            h5f.create_group("parameters")
            h5f.create_group("dynamics/it0")

            para_list = []
            para_list += [("/parameters/N", self.N)]
            para_list += [("/parameters/K12", self.K12)]
            para_list += [("/parameters/T", self.T)]
            para_list += [("/parameters/dt", self.dt)]
            para_list += [("/parameters/Lx", self.Lx)]
            para_list += [("/parameters/gamma", self.gamma)]
            para_list += [("/parameters/mov_coef", self.mov_coef)]

            for element in para_list:
                h5f.create_dataset(element[0],data=element[1])

            #h5f.create_dataset("dynamics/it0/act_mat", data = self.act_mat, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)
            #h5f.create_dataset("dynamics/it0/pos_mat", data = self.pos_mat, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)
            #h5f.create_dataset("dynamics/it0/neigh_mat", data = self.neigh_mat, compression= "gzip", dtype = np.int32, chunks = True, compression_opts = 6)

            h5f.create_dataset("dynamics/it0/act_mat", data = self.act_mat[:,::2], compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)
            h5f.create_dataset("dynamics/it0/pos_mat", data = self.pos_mat[:,:,::2]%self.Lx, compression= "gzip", dtype = np.single, chunks = True, compression_opts = 6)
            h5f.create_dataset("dynamics/it0/neigh_mat", data = self.neigh_mat[:,:,::2], compression= "gzip", dtype = np.int32, chunks = True, compression_opts = 6)

    @staticmethod
    def box_muller_noise():
        U1 = np.random.uniform()
        U2 = np.random.uniform()
        R = np.sqrt(-2 * np.log(U1))
        phi = 2 * np.pi * U2
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        return (X,Y)

class ActiveKuramoto:
    def __init__(self, N=100, K1 = 1.0 , K2 = 1.0, v = 1.0, Lx = 5, gamma = 10, dt = 0.01, T = 150):
        self.N = N
        self.Lx = Lx #In units of ell = 2v/K2
        self.K1 = K1
        self.K2 = K2
        self.v = v
        self.gamma = gamma
        self.dt = dt
        self.T = T
        self.t = np.linspace(0, self.T, int(self.T / self.dt))
        self.r, self.theta = self.initialize()
        self.pos_mat = np.zeros((N, 2, len(self.t) + 1))
        self.pos_mat[:, :, 0] = self.r
        self.act_mat = np.zeros((N,len(self.t)+1))
        self.neigh = self.find_neighbors()

    def initialize(self):
        r0 = np.random.rand(self.N, 2) * self.Lx
        theta0 = np.random.uniform(-np.pi, np.pi, self.N)
        return r0, theta0

    def find_neighbors(self):
        neighbors = np.zeros((self.N, 2))
        for i in range(self.N):
            dr = self.r - self.r[i]
            dr -= self.Lx * np.round(dr / self.Lx)
            dist = np.linalg.norm(dr, axis=1)
            dist[i] = np.inf
            nearest = np.argpartition(dist, 2)[:2]
            neighbors[i] = nearest[np.argsort(dist[nearest])]
        return neighbors

    def move(self, r, th):
        new_r = np.copy(r)
        for i in range(self.N):
            new_r[i] += [self.dt * self.v * np.cos(th[i]), self.dt * self.v* np.sin(th[i])]
        return new_r%self.Lx

    def kuramoto_global(self,th):
        new_th = np.copy(th)
        for i in range(self.N):
            pairwise = np.sum(np.sin(th - th[i]))
            pairwise *= self.K1 / self.N
            threewise = 0
            for j in range(self.N):
                for l in range(self.N):
                    if j != i and l != i:
                        threewise += np.sin(2 * th[j] - th[l] - th[i]) + np.sin(2 * th[l] - th[j] - th[i])
            threewise *= self.K2 / self.N**2
            new_th[i] += self.dt * (pairwise + threewise)
        return new_th

    def kuramoto_neigh(self, th, neigh):
        new_th = np.copy(th)
        for i in range(self.N):
            th_l = th[int(neigh[i, 0])]
            th_r = th[int(neigh[i, 1])]
            new_th[i] += self.dt * (self.K1/2 * (np.sin(th_l - th[i]) + np.sin(th_r - th[i])) + self.K2/2*(np.sin(2 * th_r - th_l - th[i]) + np.sin(2 * th_l - th_r - th[i])))
        return new_th

    def run(self):
        for i, _ in enumerate(self.t):
            self.theta = self.kuramoto_neigh(self.theta, self.neigh)
            self.r = self.move(self.r, self.theta)
            self.neigh = self.find_neighbors()
            self.pos_mat[:, :, i+1] = self.r
            self.act_mat = np.column_stack((self.act_mat, self.theta))
        return self.act_mat, self.pos_mat

#No movement in Kuramoto Global since neighbors are not changing.
    def run_global(self):
        for i, _ in enumerate(self.t):
            self.theta = self.kuramoto_global(self.theta)
            self.pos_mat[:, :, i+1] = self.r
            self.act_mat = np.column_stack((self.act_mat, self.theta))
        return self.act_mat, self.pos_mat

    def reset(self):
        self.r, self.theta = self.initialize()
        self.pos_mat = np.zeros((self.N, 2, len(self.t) + 1))
        self.pos_mat[:, :, 0] = self.r
        self.act_mat = np.reshape(self.theta, (self.N, 1))
        self.neigh = self.find_neighbors()

    @staticmethod
    def box_muller_noise():
        U1 = np.random.uniform()
        U2 = np.random.uniform()
        R = np.sqrt(-2 * np.log(U1))
        phi = 2 * np.pi * U2
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        return (X,Y)
