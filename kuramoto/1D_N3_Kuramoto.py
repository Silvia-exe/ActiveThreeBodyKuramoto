#1D, N=3 hard-coded Kuramoto

import numpy as np
import matplotlib.pyplot as plt
import plotting as kplt
from scipy.integrate import odeint
from kuramoto import Kuramoto

def epsilon_i():
    return 0.01*2*np.pi*np.random.uniform(-1,1)

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

def derivative(angles,t):
    dxdt = nat_freq + 1/2*K1*pairwise_interactions(angles) + 1/2*K2*threewise_interactions(angles)
    #dxdt -= dxdt[0]
    #print(dxdt)
    return dxdt

def integrate(angles,t):
    timeseries = odeint(derivative, angles, t)
    return timeseries.T

N = 3
T = 10
dt = 0.1
t = np.linspace(0,T,int(T/dt))
nat_freq = [0,0,0]
init_angle= 2*np.pi/3

K1 = 1
K2_vec = [2] #[0,1,2,5,10,50] #1000

#init_angles = np.array([2*np.pi*np.random.uniform(-1,1) for _ in range(N)])

#Theta1 will be used as reference moving frame
init_angles = [0+epsilon_i(), init_angle+epsilon_i(), -init_angle+epsilon_i()]
#print("Initial angles: ", init_angles)

runs = []
for k2 in K2_vec:
    K2 = k2
    act_mat = integrate(init_angles,t)
    runs.append(act_mat)

runs_array = np.array(runs)

for i, coupling in enumerate(K2_vec):
    plt.plot(
        [Kuramoto.phase_coherence(vec)
         for vec in runs_array[i, ::].T],
        label="$K_2 =$ "+ str(coupling) # higher -> darker
    )
    #print(str(coupling/np.max(K2_vec)))


plt.ylabel(r'Order parameter ($R_t$)')
plt.xlabel('Time [t]')
plt.title("Evolucion de parametro de orden")
plt.legend()
plt.show()


kplt.animate_oscillators(act_mat)
