import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, odeint

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

u0 = 1  # initial angular displacement
v0 = 1  # initial angular velocity

rs = 10 * R  #
ksiShock = rs / R
v1 = -1 / 7 * (2 * G * M / R) ** (1 / 2) * ksiShock ** (-1 / 2)
u1 = -3 / 4 * s * (G * M / R) * ksiShock ** (-4) / v1


def f(y, ksi):
    u = y[0]
    v = y[1]
    return [3 * s * G * M / R * ksi ** (-5) / v,
            gamma * v - 3 * v / ksi - 9 / 4 * s * G * M / R * ksi ** (-5) / u]


'''   derivs = [3 * s * G * M / R * ksi ** (-5) / v,  # list of dy/dt=f functions
              gamma * v - 3 * v / ksi - 9 / 4 * s * G * M / R * ksi ** (-5) / u]'''


def bc(y0, y1):
    # return np.array([y0[0] - 1, y0[1] - 1, y1[0] + 1 / 7 * (2 * G * M / R) ** (1 / 2) * ksiShock ** (-1 / 2),
    # y1[1] + -3 / 4 * s * (G * M / R) * ksiShock ** (-4) / v1])
    return [y0[0] - 1, y0[1] - 1, y1[0] + 1 / 7 * (2 * G * M / R) ** (1 / 2) * ksiShock ** (-1 / 2),
            y1[1] + -3 / 4 * s * (G * M / R) * ksiShock ** (-4) / v1]


# Bundle initial conditions for ODE solver
y0 = [u0, v0]
y1 = [u1, v1]
# Make time array for solution


ksiStop = ksiShock
ksiInc = 0.05
ksi = np.arange(1., ksiStop, ksiInc)

# Call the ODE solver
y = np.ones((2, ksi.shape[0]))
res = solve_bvp(f, bc, ksi, y, verbose=2)

# Plot results
fig = plt.figure(1, figsize=(8, 8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(ksi, res[:, 0])
ax1.set_xlabel('ksi')
ax1.set_ylabel('u')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(ksi, res[:, 1])
ax2.set_xlabel('ksi')
ax2.set_ylabel('v')

# Plot omega vs theta

plt.tight_layout()
plt.show()
