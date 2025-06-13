import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
import plotting as kplt
import numpy as np
import h5py
import math

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

class ActiveKuramotoAdim:
    def __init__(self, N=100, K12 = 1.0 , Lx = 5, gamma = 10, dt_adim = 0.01, T_adim = 150):
        self.N = N
        self.Lx = Lx #In units of ell = 2v/K2
        self.K12 = K12
        self.gamma = gamma
        self.dt = dt_adim
        self.T = T_adim
        self.t_adim = np.linspace(0, self.T, int(self.T / self.dt))
        self.r, self.theta = self.initialize()
        self.pos_mat = np.zeros((N, 2, len(self.t_adim) + 1))
        self.pos_mat[:, :, 0] = self.r
        self.act_mat = np.reshape(self.theta, (N, 1))
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
            new_r[i] += [self.dt * np.cos(th[i]), self.dt * np.sin(th[i])]
        return new_r%self.Lx

    def move_noise_WIP(self, r, th):
        new_r = np.copy(r)
        for i in range(self.N):
            nx, ny = self.box_muller_noise()
            dx = self.v*self.dt*np.cos(th[i]) + (2*1/gamma*self.dt) * nx
            dy = self.v*self.dt*np.sin(th[i]) + (2*1/gamma*self.dt) * ny
            #dx = v*dt + 1/gamma * nx
            #dy = v*dt + 1/gamma * ny
            new_r[i] += [dx, dy]
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
            new_th[i] += self.dt * (self.K12 * (np.sin(th_l - th[i]) + np.sin(th_r - th[i])) + (np.sin(2 * th_r - th_l - th[i]) + np.sin(2 * th_l - th_r - th[i])))
        return new_th

    def run(self):
        for i, _ in enumerate(self.t_adim):
            self.theta = self.kuramoto_neigh(self.theta, self.neigh)
            self.r = self.move(self.r, self.theta)
            self.neigh = self.find_neighbors()
            self.pos_mat[:, :, i+1] = self.r
            self.act_mat = np.column_stack((self.act_mat, self.theta))
        return self.act_mat, self.pos_mat

    def reset(self):
        self.r, self.theta = self.initialize()
        self.pos_mat = np.zeros((self.N, 2, len(self.t_adim) + 1))
        self.pos_mat[:, :, 0] = self.r
        self.act_mat = np.reshape(self.theta, (self.N, 1))
        self.neigh = self.find_neighbors()

    def save_data(self, filename):
        with h5py.File(filename, "w") as h5f:
            h5f.create_group("parameters")
            h5f.create_group("dynamics")

            para_list = []
            para_list += [("/parameters/N", self.N)]
            para_list += [("/parameters/K12", self.K12)]
            para_list += [("/parameters/T", self.T)]
            para_list += [("/parameters/dt", self.dt)]
            para_list += [("/parameters/Lx", self.Lx)]
            para_list += [("/parameters/gamma", self.gamma)]

            for element in para_list:
                h5f.create_dataset(element[0],data=element[1])

            h5f.create_dataset("dynamics/act_mat", data = self.act_mat, compression = 'gzip')
            h5f.create_dataset("dynamics/pos_mat", data = self.pos_mat, compression = 'gzip')

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
        self.dt = dt_adim
        self.T = T_adim
        self.t_adim = np.linspace(0, self.T, int(self.T / self.dt))
        self.r, self.theta = self.initialize()
        self.pos_mat = np.zeros((N, 2, len(self.t_adim) + 1))
        self.pos_mat[:, :, 0] = self.r
        self.act_mat = np.reshape(self.theta, (N, 1))
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
        for i, _ in enumerate(self.t_adim):
            self.theta = self.kuramoto_global(self.theta)
            self.pos_mat[:, :, i+1] = self.r
            self.act_mat = np.column_stack((self.act_mat, self.theta))
        return self.act_mat, self.pos_mat

    def reset(self):
        self.r, self.theta = self.initialize()
        self.pos_mat = np.zeros((self.N, 2, len(self.t_adim) + 1))
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
