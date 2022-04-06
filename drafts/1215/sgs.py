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

MSun = 2 * 10 ** 33  # масса молнца г
G = 6.67 * 10 ** (-8)  # гравитационная постоянная см3·с−2·г−1
M = 1.5 * MSun  # масса нз г
R = 10 ** 6  # радиус нз см

c = 3 * 10 ** 10  # скорость света см/с
l0 = 2 * 10 ** 5  # длина аккреции на поверхности м  взял как в статье
d0 = 5 * 10 ** 3  # ширина аккреции на поверхности м   взял как в статье стр 4

'''l0 = 0.5 * R  # длина аккреции на поверхности м  взял как в статье
d0 = 0.0344 * R'''

H = 10 ** 13  # магнитное поле стр 19 над формулой 37
u0 = 3 * H ** 2 / 8 / np.pi  # значение плотности излученяи на поверхности

# s = 2 * 10 ** 18 / (l0 * d0)  # поток массы M* взял 10**18 г/с эдингтоновский темп
# мб нужно умножить на 2 если 2 полюса


sigmaT = 6.652 * 10 ** (-25)  # сечение томсона см-2
massP = 1.67 * 10 ** (-24)  # масса протона г

kT = 3.9832335 * 10 ** (-1)  # томсоновская непрозрачность
k = 3.977 * 10 ** (-1)  # opacity непрозрачность взял томсоновскую  стр 12

s = c * R / k / d0 ** 2  # поток массы при условии что gamma =1

L2zv = 2 * l0 / d0 * c / k * G * M  # предельная светимость L** стр 7 формула 3
# Lt = 2 * L2zv  # светимость аккреции
Lt = 2 * l0 * d0 * G * M / R * s  # светимость аккреции
gamma = L2zv / Lt  # параметр отнаошение темпов аккреции

rs = 4.6 * R  #
ksiShock = rs / R

v1 = -1 / 7 * (2 * G * M / R) ** (1 / 2) * ksiShock ** (-1 / 2)
u1 = -3 / 4 * s * (G * M / R) * ksiShock ** (-4) / v1  # зависит от n размера пространства !!! взял n=3 везде

# Bundle parameters for ODE solver
params = [gamma, s, G, M, R]

# Bundle initial conditions for ODE solver
y0 = [u1, v1]

# Make time array for solution
ksiStop = 10.
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

a = 7.5657 * 10 ** (-15)  # радиационная константа p=aT**4  эрг см-3 К-4
u2 = np.append(psoln1[::-1, 0], psoln[:, 0])
T = (u2 / a) ** (1 / 4)

e = c / (k * s * d0)  # формула 18 стр 14


def fTeta():
    u = np.append(psoln1[::-1, 0], psoln[:, 0])
    v = np.append(psoln1[::-1, 1], psoln[:, 1])
    x = np.append(ksi1[::-1], ksi)
    return -2 / 3 * e * x ** (3 / 2) * u * v


sigmStfBolc = 5.67 * 10 ** (-5)  # постоянная стефана больцмана в сгс

Teff = (fTeta() / sigmStfBolc) ** (1 / 4)

# Plot

ax3 = fig.add_subplot(311)
ax3.plot(np.append(ksi1[::-1], ksi), np.log10(Teff), 'b')
ax3.plot(np.append(ksi1[::-1], ksi), np.log10(T), 'r')
ax3.set_xlabel('ksi')
ax3.set_ylabel('log10T')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(np.append(ksi1[::-1], ksi), np.append(psoln1[::-1, 1], psoln[:, 1]) / c)
ax2.set_xlabel('ksi')
ax2.set_ylabel('v/c')

# Plot omega vs theta

plt.tight_layout()
plt.show()

print(gamma)
