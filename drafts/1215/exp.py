import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt

special.expn(3, np.arange(1.0, 4.0, 0.5))

MSun = 2 * 10 ** 33  # масса молнца г
G = 6.67 * 10 ** (-8)  # гравитационная постоянная см3·с−2·г−1
M = 1.5 * MSun  # масса нз г
R = 10 ** 6  # радиус нз см

c = 3 * 10 ** 10  # скорость света см/с
l0 = 2 * 10 ** 5  # длина аккреции на поверхности м  взял как в статье
d0 = 5 * 10 ** 3  # ширина аккреции на поверхности м   взял как в статье стр 4

H = 10 ** 13  # магнитное поле стр 19 над формулой 37

u0 = 3 * H ** 2 / 8 / np.pi  # значение плотности излученяи на поверхности
gamma = 1  # параметр отношение темпов аккреции
k = 3.977 * 10 ** (-1)  # opacity непрозрачность взял томсоновскую  стр 12
s = c * R / k / d0 ** 2  # поток массы при условии что gamma =1
Lt = 2 * l0 * d0 * G * M / R * s  # светимость аккреции

# eta = (8 / 21 * u0 * d0 ** 2 * k / (c * (2 * G * M * R) ** (1 / 2))) ** (1 / 4)  # константа
eta = 18  # взял из 19 стр над 37 формулой


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
ksiShock = ksi2  # rs/R - находим радиус ударной волны

betta = 1 - gamma * np.exp(gamma) * (special.expn(1, gamma) - special.expn(1, gamma * ksiShock))


def u(ksi):
    return u0 * (1 - np.exp(gamma) / betta * (special.expn(2, gamma) - special.expn(2, gamma * ksi) / ksi)) ** 4


def v(ksi):
    return (3 / 4 * s * G * M / R * np.exp(gamma * ksi) / ksi ** 3 * (
            1 / ksi * special.expn(2, gamma * ksi) + betta * np.exp(-gamma) - special.expn(2, gamma))) / -u(ksi)


print(ksiShock)

ksiStop = 10.
ksiInc = 0.05
ksi = np.arange(1., ksiStop, ksiInc)


a = 7.5657 * 10 ** (-15)  # радиационная константа p=aT**4  эрг см-3 К-4
T = (u(ksi) / a) ** (1 / 4)

# Plot results
fig = plt.figure(1, figsize=(8, 8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(ksi, np.log10(T))
ax1.set_xlabel('ksi')
ax1.set_ylabel('log10T')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(ksi, v(ksi)/c)
ax2.set_xlabel('ksi')
ax2.set_ylabel('v/c')

plt.tight_layout()
plt.show()
