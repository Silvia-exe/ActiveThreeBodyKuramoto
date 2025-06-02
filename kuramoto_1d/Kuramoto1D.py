
import numpy as np
import matplotlib.pyplot as plt
import plotting as kplt
from scipy.integrate import odeint

#Returns a small random number between 0.01 * (-pi, pi)
def epsilon_i():
    return 0.01*np.random.uniform(-np.pi,np.pi, N)

#Interactions with immediate neighbors
def pairwise_interactions(angles):
    interactions = np.zeros(N)
    for i in range(N):
        interactions[i] = np.sin(angles[(i-1)%N]-angles[i]) + np.sin(angles[(i+1)%N]-angles[i])
    return interactions

#Interactions with triangle
def threewise_interactions(angles):
    interactions=np.zeros(N)
    for i in range(N):
        interactions[i] =  np.sin(2*angles[(i+1)%N]-angles[(i-1)%N]-angles[i]) + np.sin(2*angles[(i-1)%N]-angles[(i+1)%N]-angles[i])
    return interactions

#Derivative to be solved, inertial frame of reference
def derivative(angles,t):
    dxdt = nat_freq + 1/2*K1*pairwise_interactions(angles) + 1/2*K2*threewise_interactions(angles)
    dxdt -= dxdt[0]
    #print(dxdt)
    return dxdt

#Integration function for Kuramoto system
def integrate(angles,t):
    timeseries = odeint(derivative, angles, t)
    return timeseries.T

N = 6 #Number of nodes
T = 10 #Total integration time
dt = 0.01 #Time step
t = np.linspace(0,T,int(T/dt)) #Timesteps to integrate
nat_freq = np.zeros(N) #Natural frequencies of oscillators
pert = epsilon_i() #Vector of small angle perturbations

K1 = 1
K2 = 3

#init_angles = np.array([2*np.pi*np.random.uniform(-1,1) for _ in range(N)]) #Random initial angles

#init_angles = [0,np.pi/3,2*np.pi/3,3*np.pi/3,4*np.pi/3,5*np.pi/3]
init_angles = [0,2*np.pi/3,4*np.pi/3,6*np.pi/3,8*np.pi/3,10*np.pi/3]
init_pert = init_angles+pert

print("Initial angles: ", init_pert)

act_mat = integrate(init_pert,t)
print("Final angles:", act_mat.T[-1])

kplt.plot_phase_coherence(act_mat)
kplt.oscillators_comp(act_mat)
kplt.animate_oscillators(act_mat)
