import numpy as np
import random as rnd
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#2D directed random walk of N particles

#Non-directed brownian motion, no boundary conditions, as control. WIP
def true_brownian(v1):
    r0b = np.array([(0,0) for _ in range(N)])
    r_v = np.array([r0b])
    T_vb = np.arange(0, T,step = dt)
    pos_vb = np.array([r0b]) #Positions of particles

    r_oldb = r0b
    for i in range(0,T):
        r_oldb +=  np.array([r + v1*rnd.uniform(-1,1)*dt for r in r_oldb])
        r_v = np.append(r_v, r_oldb, axis = 0)
        true_brownian.append(r_old)
    for p in range(N):
        plt.plot(pos_vb[:, p, 0], pos_vb[:, p, 1])
    plt.xlabel("x-axis [x]")
    plt.ylabel("y-axis [y]")
    plt.title("2D true Brownian motion of "+ str(N) + " particles")
    plt.show()

#Implements reflectibe boundary conditions.
def reflective_bc(delta):
    if v0 == 0:
        return 0

    diff = abs(delta)//v0
    delta += -1*np.sign(delta)*abs(diff)*v0 #Periodic boundary conditions
    #delta *= 1-2*(abs(diff)%2) #Reflection?
    return delta

#Acceptance function for movement
#Will stop a particle if it would intersect another
def p_acceptance(r_old, r_new):
    for p in range(N):
        for k in range(N):
            if (np.sqrt((r_new[p,0] - r_old[k,0])**2 + (r_new[p,1] - r_old[k,1])**2)) < 2*sigma:
                if k != p:
                    r_new[p] = r_old[p]
    return r_new

#Updates the position of the particle
def update_r(r_old, del_old, v1):
    del_new = np.array([reflective_bc(d + v1*rnd.uniform(-1,1)) for d in del_old])
    r_new = np.array([r_old[i] + del_new[i]*dt for i in range(len(del_old))])
    return (r_new, del_new)

#Initializes N particles with radius sigma such that they are not overlapped
def initialize_r0():
    r0_v = np.random.rand(1,2)*Lx
    for i in range(N-1):
        placed = False
        while not placed:
            r0_try = np.random.rand(1,2)*Lx
            if all(math.hypot(r0_v[p,0]-r0_try[0,0], r0_v[p,1]-r0_try[0,1]) >= sigma*2 for p in range(len(r0_v))):
                r0_v = np.append(r0_v, r0_try, axis = 0)
                placed = True
    return r0_v

#Detects if particle| is at boundary and implements periodic boundary conditions
def periodic_bc(r_new):
    #Detect boundary crossing and implement PBC
    for p in range(N):
        for i in range(2):
            if r_new[p,i] < 0:
                r_new[p,i] += Lx
            elif r_new[p,i] > Lx:
                r_new[p,i] -= Lx

    return r_new

def animate_positions(r_v):
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lx)
    ax.set_aspect('equal', adjustable = "box")

    # Initialize circle artists
    circles = [plt.Circle((0, 0), sigma, color='blue', alpha=0.6) for _ in r_v[0]]
    for c in circles:
        ax.add_patch(c)

    def update(frame_idx):
        positions = r_v[frame_idx]
        for circle, pos in zip(circles, positions):
            circle.center = pos
        return circles

    ani = animation.FuncAnimation(fig, update, frames=len(r_v), interval=30, blit=True)
    plt.show()
    return ani

def MSD2D(r_v):
    T_len, p_len, _ = r_v.shape
    msd = np.zeros(T_len)
    print(r_v[0])
    for dt in range(T_len):
        print(dt)
        disp = r_v[0,:] - r_v[dt,:]
        print(disp)
        #sqrd_disp = np.sum(disp**2, axis = 2)
        #msd[dt] = np.mean(sqrd_disp)

    return msd

def main():
    #Simulation variables
    global T
    T = 10 #Total simulation time
    global dt
    dt = 0.1 #dt for T
    global N
    N = 10 #Number of particles
    global sigma
    sigma = 1 #Particle radius and system length scale
    global Lx
    Lx = 50*sigma #Box size

    global v0
    v0 = 1.0 #Directed velocity, will describe self-propelled motion
    tau_v = [1.0] #Persistance time

    #Figure settings
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lx)
    plt.gca().set_aspect('equal', adjustable='box')

    #Position update loop for each specified tau
    for tau in tau_v:
        v1 = (v0*dt)/np.sqrt(tau) #Random drift velocity, amplitude of random displacement

        r0 = initialize_r0() #Initializes particles (or hard disks) without overlap

        del0 = reflective_bc(np.random.uniform(-1,1,(N,2))*v0) #Random initial delta

        T_v = np.arange(0, T,step = dt) #Time vector

        r_v = np.array([r0]) #Positions of particles
        del_v = np.array([del0]) #Random fluctuations of particles

        #posdiff_v = np.array([r0]) #For MSD WIP

        r_old = r0
        del_old = del0

        #shows initial position
        circles = [plt.Circle(p, sigma, color='blue', alpha=0.6) for p in r0]
        for circle in circles:
            ax.add_patch(circle)
        plt.show()

        #Loop for all positions in T
        for t in range(len(T_v)-1):
            #Positions are proposed and saved as r_new.
            r_new, del_new = update_r(r_old, del_old,v1)

            #If a particle is out of bounds, forces periodic boundary conditions
            r_new = periodic_bc(r_new)

            #The new proposed positions are compared to the old positions. Positions are accepted sequentially if no overlap happens
            #Done at the end!
            r_new = p_acceptance(r_old, r_new)

            #Saving of particle positions in position vectors r_v and old
            r_v = np.append(r_v, [r_new], axis = 0)
            del_v = np.append(del_v, [del_new], axis = 0)

            r_old = r_new
            del_old = del_new

        ani = animate_positions(r_v)
        #ani.save("tau_"+str(tau)+"_v0_"+str(v0)+"_dt_"+str(dt)+".gif", writer="pillow", fps=30, dpi=200)

        #msd = MSD2D(r_v)

    '''plt.plot(msd)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Time [t]")
    plt.legend()
    plt.ylabel("MSD [$x^2$] (Ensemble average)")
    plt.title("Mean Square Displacement")
    plt.show()'''

    #pos_v[:, 0] #x and y of 0th particle
    #pos_v[:, 1, 0] #x of pth particle'''

    '''for p in range(N):
        plt.plot(r_v[:, p, 0], r_v[:, p, 1])
    plt.xlabel("x-axis [x]")
    plt.ylabel("y-axis [y]")
    #plt.legend()
    plt.title("2D directed Brownian motion of "+ str(N) + " particles")
    plt.show()'''


    '''plt.plot(T_v, del_v[:, 0], marker='.')
    plt.xlabel("Time [t]")
    plt.ylabel("$\delta(t)$ [x]")
    plt.title("$\delta(t)$")
    plt.show()'''

if __name__ == "__main__":
    main()
