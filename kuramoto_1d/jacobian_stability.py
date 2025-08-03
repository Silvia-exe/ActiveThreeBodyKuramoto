import sympy as sp
import numpy as np

#Defining symbols
th = sp.symbols('th1:7') # th1 to th6
w = sp.symbols('w1:7') # w1 to w6
K, K1, K2 = sp.symbols('K K1 K2') #Coupling parameters
phi2, phi3 = sp.symbols('phi2 phi3') #For N=3 nodes, reduced frame of reference
lam = sp.Symbol('lambda') #For eigenvalues

#Defining variables
variables_movfr = [phi2, phi3] #ph2 = th2-th1 and ph3 = th3-th1

#Number of nodes
N = 3

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
]'''

f_movfr_N3 = [
    w[2] - w[1] + K1/2*(sp.sin(phi3-phi2)-2*sp.sin(phi2)-sp.sin(phi3)) + K2/2*(sp.sin(phi3-2*phi2)-sp.sin(phi2+phi3)),
    w[3] - w[1] + K1/2*(sp.sin(phi2-phi3)-2*sp.sin(phi3)-sp.sin(phi2)) + K2/2*(sp.sin(phi2-2*phi3)-sp.sin(phi3+phi2))
]

f_movfr_N3_adim = [
    w[2] - w[1] + K*(sp.sin(phi3-phi2)-2*sp.sin(phi2)-sp.sin(phi3)) + (sp.sin(phi3-2*phi2)-sp.sin(phi2+phi3)),
    w[3] - w[1] + K*(sp.sin(phi2-phi3)-2*sp.sin(phi3)-sp.sin(phi2)) + (sp.sin(phi2-2*phi3)-sp.sin(phi3+phi2))
]

#Symbolic Jacobian
f_N6 = function_vect(3)
#J = sp.Matrix(f_N6).jacobian(th)
J = sp.Matrix(f_movfr_N3_adim).jacobian((phi2,phi3))
sp.pprint(J)

#Substituting variables
subs = {
    phi2: np.pi/2,
    phi3: 0,
    w[0]: 0,
    w[1]: 0,
    w[2]: 0,
    K:1.0 
}

#Defining characteristic matrix and printing characteristic equation
#char_matrix = J - lam * sp.eye(2)
#char_poly = char_matrix.det()
#sp.pprint(char_poly)

#Eval. the jacobian for the eigenvalues
J_eval = J.subs(subs)
sp.pprint(J_eval)
trace = J_eval.trace()
determinant = J_eval.det()
eigenvalues = J_eval.eigenvals()
print("Trace:", trace)
print("Determinant:", determinant)
print("Eigenvectors:", J_eval.eigenvects())
#print(np.array(list(eigenvalues.items())))
