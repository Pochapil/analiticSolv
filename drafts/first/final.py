import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def f(y, ksi, params):
    u, v = y  # unpack current values of y
    gamma, s, G, M, R = params  # unpack parameters
    derivs = [3 * s * G * M / R * ksi ** (-5) / v,  # list of dy/dt=f functions
              gamma * v - 3 * v / ksi - 9 / 4 * s * G * M / R * ksi ** (-5) / u]
    return derivs


# Parameters

MSun = 2 * 10 ** 30  # масса молнца кг
G = 6.67 * 10 ** (-11)  # гравитационная постоянная
M = 1.5 * MSun  # масса нз кг
R = 10 ** 4  # радиус нз м

c = 3 * 10 ** 8  # скорость света м/с
'''l0 = 2 * 10 ** 3  # длина аккреции на поверхности м  взял как в статье
d0 = 5 * 10  # ширина аккреции на поверхности м   взял как в статье стр 4'''

l0 = 0.5 * R  # длина аккреции на поверхности м  взял как в статье
d0 = 0.0344 * R

s = 2 * 10 ** 15 / (l0 * d0)  # поток массы

k = 3.977 * 10 ** (-2)  # opacity непрозрачность взял томсоновскую  стр 12
L2zv = 2 * l0 / d0 * c / k * G * M  # предельная светимость L**
Lt = 2 * L2zv  # светимость аккреции

gamma = L2zv / Lt  #

# Initial values
rs = 4.6 * R  #
ksiShock = rs / R

v1 = -1 / 7 * (2 * G * M / R) ** (1 / 2) * ksiShock ** (-1 / 2)
u1 = -3 / 4 * s * (G * M / R) * ksiShock ** (-4) / v1

# Bundle parameters for ODE solver
params = [gamma, s, G, M, R]

# Bundle initial conditions for ODE solver
y0 = [u1, v1]

# Make time array for solution
ksiStop = 20.
ksiInc = 0.05
ksi = np.arange(ksiShock, ksiStop, ksiInc)

# Call the ODE solver
psoln = odeint(f, y0, ksi, args=(params,), mxstep=5000000)

ksiStop1 = 1.
ksiInc1 = -0.05
ksi1 = np.arange(ksiShock, ksiStop1, ksiInc1)

psoln1 = odeint(f, y0, ksi1, args=(params,), mxstep=5000000)

# Plot results
fig = plt.figure(1, figsize=(8, 8))

# Plot theta as a function of time

a = 7.5657 * 10 ** (-16)
u2 = np.append(psoln1[::-1, 0], psoln[:, 0])
T = (u2 / a) ** (1 / 4)

ax3 = fig.add_subplot(311)
ax3.plot(np.append(ksi1[::-1], ksi), np.log10(T))
ax3.set_xlabel('ksi')
ax3.set_ylabel('T')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(np.append(ksi1[::-1], ksi), np.append(psoln1[::-1, 1], psoln[:, 1]) / c)
ax2.set_xlabel('ksi')
ax2.set_ylabel('v/c')

# Plot omega vs theta

plt.tight_layout()
plt.show()
