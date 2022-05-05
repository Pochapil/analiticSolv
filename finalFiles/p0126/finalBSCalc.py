import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.special as special


# 30 формула
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
H = 10 ** 13  # магнитное поле стр 19 над формулой 37

c = 3 * 10 ** 10  # скорость света см/с
l0 = 2 * 10 ** 5  # длина аккреции на поверхности м  взял как в статье
d0 = 5 * 10 ** 3  # ширина аккреции на поверхности м   взял как в статье стр 4

u0 = 3 * H ** 2 / 8 / np.pi  # значение плотности излучения на поверхности

# s = 2 * 10 ** 18 / (l0 * d0)  # поток массы M* взял 10**18 г/с эдингтоновский темп
# мб нужно умножить на 2 если 2 полюса

# gamma = 1  # параметр отношение темпов аккреции

sigmaT = 6.652 * 10 ** (-25)  # сечение томсона см-2
massP = 1.67 * 10 ** (-24)  # масса протона г

# kT = 3.9832335 * 10 ** (-1)  # томсоновская непрозрачность
kT = 0.4
k = 3.977 * 10 ** (-1)  # opacity непрозрачность взял томсоновскую  стр 12 (под 11 формулой ?)

s = c * R / k / d0 ** 2  # поток массы при условии что gamma =1 (мб из 3 формулы ?) ур стр 14 ур 18 и стр 17 под ур 30

L2zv = 2 * l0 / d0 * c / k * G * M  # предельная светимость L** стр 7 формула 3
# Lt = 2 * L2zv  # светимость аккреции
Lt = 2 * l0 * d0 * G * M / R * s  # светимость аккреции 13 стр формула 14
gamma = L2zv / Lt  # параметр отнаошение темпов аккреции
# gamma = 1
# eta = (8 / 21 * u0 * d0 ** 2 * k / (c * (2 * G * M * R) ** (1 / 2))) ** (1 / 4)  # константа
# eta = 18  # взял из 19 стр над 37 формулой
eta = 16.7477  # взял из сообщения в телеге


# gamma = 1  # параметр отношение темпов аккреции

# 34 формула
def findKsiShock():
    def f(x):
        return eta * gamma ** (1 / 4) * x ** (7 / 8) - 1 - np.exp(gamma * x) * \
               (x * special.expn(2, gamma) - special.expn(2, gamma * x))

    def df(x):
        return 7 / 8 * eta * gamma ** (1 / 4) * x ** (-1 / 8) - gamma * np.exp(gamma * x) * \
               (x * special.expn(2, gamma) - special.expn(2, gamma * x)) - np.exp(gamma * x) * \
               (special.expn(2, gamma) + gamma * special.expn(1, gamma * x))

    def nuton(x):
        return x - f(x) / df(x)

    delta = 0.001  # точность для метода ньютона
    ksi1 = 4.3 * R / R
    ksi2 = nuton(ksi1)
    while np.abs((ksi1 - ksi2)) > delta:
        ksi1 = ksi2
        ksi2 = nuton(ksi1)
    return ksi2  # rs/R - находим радиус ударной волны


ksiShock = findKsiShock()

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
psoln = odeint(f, y0, ksi, args=(params,), mxstep=5000000)  # от ксишок до 10

ksiStop1 = 1.
ksiInc1 = -0.05
ksi1 = np.arange(ksiShock, ksiStop1, ksiInc1)

psoln1 = odeint(f, y0, ksi1, args=(params,), mxstep=5000000)  # от 0 до ксишок

# analytic solve bs 35 формула
betta = 1 - gamma * np.exp(gamma) * (special.expn(1, gamma) - special.expn(1, gamma * ksiShock))


# 32 формула
def u(ksi):
    return u0 * (1 - np.exp(gamma) / betta * (special.expn(2, gamma) - special.expn(2, gamma * ksi) / ksi)) ** 4


def v(ksi):
    return (3 / 4 * s * G * M / R * np.exp(gamma * ksi) / (ksi ** 3) * (
            1 / ksi * special.expn(2, gamma * ksi) + betta * np.exp(-gamma) - special.expn(2, gamma))) / -u(ksi)


a = 7.5657 * 10 ** (-15)  # радиационная константа p=aT**4  эрг см-3 К-4
u2 = np.append(psoln1[::-1, 0], psoln[:, 0])
T = (u2 / a) ** (1 / 4)
ksibs = np.append(ksi1[::-1], ksi)
Tbs = (u(ksibs) / a) ** (1 / 4)  # настоящее аналитическое решение

e = c / (k * s * d0)  # формула 18 стр 14


# 21 стр конец файла
def fTeta():
    u = np.append(psoln1[::-1, 0], psoln[:, 0])
    v = np.append(psoln1[::-1, 1], psoln[:, 1])
    x = np.append(ksi1[::-1], ksi)
    return -2 / 3 * e * x ** (3 / 2) * u * v


# 21 стр конец файла
def fTetabs(x):
    return -2 / 3 * e * x ** (3 / 2) * u(x) * v(x)


# ro = 1  # плотность падающего газа
# Fr(ksi) 30 формула 17 стр
def fr(x):
    return 4 / 3 * u(x) * v(x) + s * G * M / R * x ** (-4)


# 19 стр под конец
def q(ksi):
    return (ksi ** 3 * fr(ksi) - ksiShock ** 3 * fr(ksiShock)) * R / (s * G * M)


def frCalc(u, v, x):
    return 4 / 3 * u * v + s * G * M / R * x ** (-4)


def qCalc(u, v, ksi):
    return (ksi ** 3 * frCalc(u, v, ksi) - ksiShock ** 3 * frCalc(u, v, ksi)) * R / (s * G * M)


sigmStfBolc = 5.67 * 10 ** (-5)  # постоянная стефана больцмана в сгс

Teff = (fTeta() / sigmStfBolc) ** (1 / 4)
Teffbs = (fTetabs(ksibs) / sigmStfBolc) ** (1 / 4)

# Plot results
fig = plt.figure(1, figsize=(8, 8))

# Plot T
ax3 = fig.add_subplot(321)
ax3.plot(np.append(ksi1[::-1], ksi), np.log10(Teff), 'b', label='Teff')
ax3.plot(np.append(ksi1[::-1], ksi), np.log10(T), 'r', label='Tin')
ax3.plot(ksibs, np.log10(Tbs), 'g', label='Tinbs')  # аналитическое решение
ax3.plot(ksibs, np.log10(Teffbs), 'y', label='Teffbs')  # аналитическое решение
ax3.set_xlabel('ksi')
ax3.set_ylabel('log10T')
ax3.legend()

# Plot
r = np.arange(1, ksiShock, 0.05)
ax2 = fig.add_subplot(325)
ax2.plot(r, q(r), 'b')
# ax2.plot(ksi1[::-1],  qCalc(psoln1[::-1, 0], psoln1[::-1, 1], ksi1[::-1]), 'r')
# ax2.plot(ksi1[::-1], psoln1[::-1, 1] * psoln1[::-1, 0], 'r')

# # plot T
# ax1 = fig.add_subplot(326)
# ax1.plot(ksi1[::-1], np.log10(Teff[:len(ksi1)]), 'b', label='Teff')
# ax1.plot(ksi1[::-1], np.log10(Teffbs[:len(ksi1)]), 'r', label='Teffbs')  # аналитическое решение
# ax1.set_xlabel('ksi')
# ax1.set_ylabel('Teff')
# ax1.legend()

ax4 = fig.add_subplot(323)
ax4.plot(ksi1[::-1], np.log10(Teff[:len(ksi1)]), 'b', label='Teff')
ax4.plot(ksi1[::-1], np.log10(T[:len(ksi1)]), 'r', label='Tin')
ax4.set_xlabel('ksi')
ax4.set_ylabel('log10T')
ax4.legend()

ax5 = fig.add_subplot(324)
ax5.plot(ksi1[::-1], np.log10(Teffbs[:len(ksi1)]), 'b', label='Teffbs')
ax5.plot(ksi1[::-1], np.log10(Tbs[:len(ksi1)]), 'r', label='Tinbs')
ax5.set_xlabel('ksi')
ax5.set_ylabel('log10Tbs')
ax5.legend()

ax6 = fig.add_subplot(322)
ax6.plot(ksi1[::-1], psoln1[::-1, 1] / c, 'b', label='v/c')
ax6.plot(ksi1[::-1], v(ksi1[::-1]) / c, 'r', label='vbs/c')
ax6.set_xlabel('ksi')
ax6.set_ylabel('v/c')
ax6.legend()

plt.tight_layout()
plt.show()

print('gamma %f' % gamma)
print("ksiShock %f" % ksiShock)
print("betta %f" % betta)
print("s %f" % s)
print('vbs: %f, v0: %f' % (v(ksiShock), v1))
