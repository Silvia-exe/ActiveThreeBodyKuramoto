import sympy as sp
import numpy as np

#Defining symbols
th = sp.symbols('th1:7') # th1 to th6
w = sp.symbols('w1:7') # w1 to w6
K1, K2 = sp.symbols('K1 K2') #Coupling parameters
phi2, phi3 = sp.symbols('phi2 phi3') #For N=3 nodes, reduced frame of reference
lam = sp.Symbol('lambda') #For eigenvalues

#Defining variables
variables_movfr = [phi2, phi3] #ph2 = th2-th1 and ph3 = th3-th1

#Number of nodes
N = 6

#Defines the N-long function vector
def function_vect(N):
    f_N = []
    for i in range(N):
        th_i = th[i]
        w_i = w[i]

        # Vecinos inmediatos (en red tipo anillo)
        th_l = th[(i - 1) % N]
        th_r = th[(i + 1) % N]

        # Término K1: acoplamientos bilineales con vecinos
        term_K1 = K1/2 *(sp.sin(th_l - th_i) + sp.sin(th_r - th_i))
        term_K2 = K2 / 2 * sp.sin(2*th_l - th_r - th_i) + sp.sin(2*th_r - th_l - th_i)

        # Ecuación total para el nodo i
        f_i = w_i + term_K1 + term_K2
        f_N.append(f_i)
    return f_N

#Defining function vectors for three body system and moving frame three body system, just in case
'''f_N3 = [
    w[0] + K1/2*(sp.sin(th2-th1) + sp.sin(th3-th1)) + K2/2*(sp.sin(2*th2 - th3 - th1) + sp.sin(2*th3 - th2 - th1)),
    w[1] + K1/2*(sp.sin(th1-th2) + sp.sin(th3-th2)) + K2/2*(sp.sin(2*th1 - th2 - th3) + sp.sin(2*th3 - th2 - th1)),
    w[2] + K1/2*(sp.sin(th1-th3) + sp.sin(th2-th3)) + K2/2*(sp.sin(2*th1 - th2 - th3) + sp.sin(2*th2 - th3 - th1))
]

f_movfr_N3 = [
    w2 - w1 + K1/2*(sp.sin(phi3-phi2)-2*sp.sin(phi2)-sp.sin(phi3)) + K2/2*(sp.sin(phi3-2*phi2)-sp.sin(phi2+phi3)),
    w3 - w1 + K1/2*(sp.sin(phi2-phi3)-2*sp.sin(phi3)-sp.sin(phi2)) + K2/2*(sp.sin(phi2-2*phi3)-sp.sin(phi3+phi2))
]'''

#Symbolic Jacobian
f_N6 = function_vect(6)
J = sp.Matrix(f_N6).jacobian(th)
#sp.pprint(J)

#Substituting variables
subs = {
    th[0]: 0,
    th[1]: np.pi/3,
    th[2]: 2*np.pi/3,
    th[3]: np.pi,
    th[4]: 4*np.pi/3,
    th[5]: 5*np.pi/3,
    w[0]: 0,
    w[1]: 0,
    w[2]: 0,
    w[3]: 0,
    w[4]: 0,
    w[5]: 0,
    K1: 1,
    K2: 10
}

#Defining characteristic matrix and printing characteristic equation
#char_matrix = J - lam * sp.eye(2)
#char_poly = char_matrix.det()
#sp.pprint(char_poly)

#Eval. the jacobian for the eigenvalues
J_eval = J.subs(subs)
eigenvalues = J_eval.eigenvals()
print(np.array(list(eigenvalues.items())))
