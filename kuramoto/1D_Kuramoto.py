#1D, N=3 hard-coded Kuramoto

import numpy as np
import matplotlib.pyplot as plt
import plotting as kplt
from scipy.integrate import odeint
from kuramoto import Kuramoto

#Returns a small random number between 0.01 * (-pi, pi)
def epsilon_i():
    return 0.01*np.random.uniform(-np.pi,np.pi)

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

def order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

def epsilon_i():
    return 0.005*2*np.pi*np.random.uniform(-1,1)

N = 6
T = 10
dt = 0.1
t = np.linspace(0,T,int(T/dt))
nat_freq = np.zeros(N)
init_angle= 2*np.pi/3

K1 = 1
K2_vec = [2] #[0,1,2,5,10,50] #1000

#init_angles = np.array([2*np.pi*np.random.uniform(-1,1) for _ in range(N)])

init_angles = [0+epsilon_i(), init_angle+epsilon_i(), -init_angle+epsilon_i()]
#print("Initial angles: ", init_angles)

init_angles = [0, 0, 4*np.pi/6+epsilon_i(), 6*np.pi/6+epsilon_i(), 8*np.pi/6+epsilon_i(), 10*np.pi/6+epsilon_i()] #np.random.uniform(-np.pi,np.pi,N)
print("Initial angles: ", init_angles)

act_mat = integrate(init_angles,t)
print("Final angles:", act_mat.T[-1])

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
