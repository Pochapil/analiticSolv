import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

s = 1  # хз что это
MSun = 2 * 10 ** 30  # масса молнца кг
G = 6.67 * 10 ** (-11)  # гравитационная постоянная
M = 1.5 * MSun  # масса нз кг
R = 10 ** 4  # радиус нз м

c = 3 * 10 ** 8  # скорость света м/с
l0 = 2 * 10 ** 3  # длина аккреции на поверхности м  взял как в статье
d0 = 5 * 10  # ширина аккреции на поверхности м   взял как в статье стр 4
k = 3.977 * 10 ** (-2)  # opacity непрозрачность взял томсоновскую стр 12
L2zv = 2 * l0 / d0 * c / k * G * M  # предельная светимость L**
Lt = 2 * L2zv  # светимость аккреции

gamma = L2zv / Lt  #

u0 = 1
# betta = -4 / 3 * u(1) * v(1) * R / (s * G * M)  # часть энергии,
# которая выделяется при аккреции, которая уносится тонущим газом под поверхность нейтронной звезды.
betta = 1 / 2  # стр 10-11


def u(ksi):
    return u0 * (1 - np.exp(gamma) / betta * (special.expn(2, gamma) - special.expn(2, gamma * ksi) / ksi)) ** 4


def v(ksi):
    return (3 / 4 * s * G * M / R * np.exp(gamma * ksi) / ksi ** 3 * (
            1 / ksi * special.expn(2, gamma * ksi) + betta * np.exp(-gamma) - special.expn(2, gamma))) / -u(ksi)


# special.expn(3, np.arange(1.0, 4.0, 0.5))  # экспонента степени n (сейчас n =3 )

ksiStop = 10.
ksiInc = 0.05
ksi = np.arange(1., ksiStop, ksiInc)

# Plot results
fig = plt.figure(1, figsize=(8, 8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(ksi, u(ksi))
ax1.set_xlabel('ksi')
ax1.set_ylabel('u')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(ksi, v(ksi))
ax2.set_xlabel('ksi')
ax2.set_ylabel('v')

plt.tight_layout()
plt.show()