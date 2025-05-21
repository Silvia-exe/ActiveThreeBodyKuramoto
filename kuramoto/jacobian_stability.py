import sympy as sp
import numpy as np


#Defining symbols
th = sp.symbols('th1:7')  # th1 to th6 -> tuple of 6 elements
w = sp.symbols('w1:7')        # w1 to w6
K1, K2 = sp.symbols('K1 K2')
phi2, phi3 = sp.symbols('phi2 phi3')
lam = sp.Symbol('lambda')

#Defining variables
variables_movfr = [phi2, phi3] #ph2 = th2-th1 and ph3 = th3-th1
f_N6 = []

# Número de nodos
N = 6
# Para nodos 0 a 5 (th1 a th6)
for i in range(N):
    th_i = th[i]
    w_i = w[i]

    # Vecinos inmediatos (en red tipo anillo)
    left = th[(i - 1) % N]
    right = th[(i + 1) % N]

    # Término K1: acoplamientos bilineales con vecinos
    term_K1 = (sp.sin(left - th_i) + sp.sin(right - th_i)) * K1 / 2

    # Término K2: combinaciones de tres cuerpos
    # Definimos los tres nodos con los que se hace la interacción de tipo K2
    k2_terms = []
    triplets = [
        ((i + 1) % N, (i - 1) % N),
        ((i + 2) % N, (i + 1) % N),
        ((i - 2) % N, (i - 1) % N)
    ]
    for j, k in triplets:
        th_j = th[j]
        th_k = th[k]
        k2_terms.extend([
            sp.sin(2*th_j - th_k - th_i),
            sp.sin(2*th_k - th_j - th_i)
        ])

    term_K2 = K2 / 6 * sum(k2_terms)

    # Ecuación total para el nodo i
    f_i = w_i + term_K1 + term_K2
    f_N6.append(f_i)

#Defining function vectors for three body system and moving frame three body system
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
J = sp.Matrix(f_N6).jacobian(th)
#sp.pprint(J)

#Substituting variables
subs = {
    th[0]: 0,
    th[1]: 0,#2*np.pi/6,
    th[2]: 0,#4*np.pi/6,
    th[3]: 0,#6*np.pi/6,
    th[4]: 0,#8*np.pi/6,
    th[5]: 0,#10*np.pi/6,
    w[0]: 0,
    w[1]: 0,
    w[2]: 0,
    w[3]: 0,
    w[4]: 0,
    w[5]: 0,
    K1: 1,
    K2: 20
}

#Defining characteristic matrix and printing characteristic equation
#char_matrix = J - lam * sp.eye(2)
#char_poly = char_matrix.det()
#sp.pprint(char_poly)

#Eval. the jacobian for the eigenvalues
J_eval = J.subs(subs)
eigenvalues = J_eval.eigenvals()
print(np.array(list(eigenvalues.items())))

#Printing and feeding eigenvalues to sorting algorithm
#print("Jacobiano evaluado en el punto fijo:")
#sp.pprint(J_eval)
#print("\nEigenvalores:")
#print(eigenvalues)
