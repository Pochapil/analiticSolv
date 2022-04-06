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
s = 1  # хз что это
MSun = 2 * 10 ** 30  # масса молнца кг
G = 6.67 * 10 ** (-11)  # гравитационная постоянная
M = 1.5 * MSun  # масса нз кг
R = 10 ** 4  # радиус нз м

c = 3 * 10 ** 8  # скорость света м/с
l0 = 2 * 10 ** 3  # длина аккреции на поверхности м  взял как в статье
d0 = 5 * 10  # ширина аккреции на поверхности м   взял как в статье стр 4
k = 3.977 * 10 ** (-2)  # opacity непрозрачность взял томсоновскую  стр 12
L2zv = 2 * l0 / d0 * c / k * G * M  # предельная светимость L**
Lt = 2 * L2zv  # светимость аккреции

gamma = L2zv / Lt  #

# Initial values
u0 = 1  # initial angular displacement
v0 = 1  # initial angular velocity

# Bundle parameters for ODE solver
params = [gamma, s, G, M, R]

# Bundle initial conditions for ODE solver
y0 = [u0, v0]

# Make time array for solution
ksiStop = 10.
ksiInc = 0.05
ksi = np.arange(1., ksiStop, ksiInc)

# Call the ODE solver
psoln = odeint(f, y0, ksi, args=(params,), mxstep=5000000)

# Plot results
fig = plt.figure(1, figsize=(8, 8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(ksi, psoln[:, 0])
ax1.set_xlabel('ksi')
ax1.set_ylabel('u')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(ksi, psoln[:, 1])
ax2.set_xlabel('ksi')
ax2.set_ylabel('v')

# Plot omega vs theta

plt.tight_layout()
plt.show()
