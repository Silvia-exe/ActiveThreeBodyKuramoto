import numpy as np
import plotting as kplt
import matplotlib.pyplot as plt

#Random data and model tests.

#Compute global order parameter R_t
def phase_coherence(angles_vec):
        suma = sum([(np.e ** (1j * i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

#Compute global order parameter R_t
def phase_coherence_pair(angles_vec):
        suma = sum([(np.e ** (1j * 2*i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

#Compute global order parameter R_t
def phase_coherence_three(angles_vec):
        suma = sum([(np.e ** (1j * 3*i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

alpha = np.pi*0.328
theta_aligned = [alpha, alpha, alpha]
theta_antialigned = [alpha, alpha + np.pi, alpha - np.pi]
theta_three = [alpha, alpha + 2*np.pi/3, alpha - 2*np.pi/3]
theta_test = [alpha, alpha + 2*np.pi/3, alpha -np.pi, alpha + np.pi*0.1927]

print("For alignment:")
print("R1 = ", phase_coherence(theta_aligned))
print("R2 = ", phase_coherence_pair(theta_aligned))
print("R3 = ", phase_coherence_three(theta_aligned))

print("For anti-alignment:")
print("R1 = ", phase_coherence(theta_antialigned))
print("R2 = ", phase_coherence_pair(theta_antialigned))
print("R3 = ", phase_coherence_three(theta_antialigned))

print("For three-alignment:")
print("R1 = ", phase_coherence(theta_three))
print("R2 = ", phase_coherence_pair(theta_three))
print("R3 = ", phase_coherence_three(theta_three))

print("For test-alignment:")
print("R1 = ", phase_coherence(theta_test))
print("R2 = ", phase_coherence_pair(theta_test))
print("R3 = ", phase_coherence_three(theta_test))

plt.plot(np.cos(theta_test),
            np.sin(theta_test),
            'o',
            markersize=10)
plt.show()
