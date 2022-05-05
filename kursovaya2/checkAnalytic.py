import geomtetricTask.matrix as matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import config

matplotlib.use("TkAgg")
import teffBsCheck as teffBsCheck

# смотри RotationSingleColorBar.py

grad_to_rad = np.pi / 180
# угол между нормалью к двойной системе и наблюдателем
i_angle = 30 * grad_to_rad
e_obs = np.array([0, np.sin(i_angle), np.cos(i_angle)])

# угол между осью вращения системы и собственным вращенеим НЗ
betta_rotate = 0 * grad_to_rad
fi_rotate = 12 * grad_to_rad

# угол между собственным вращенеим НЗ и магнитной осью
betta_mu = 15 * grad_to_rad
fi_mu = 32 * grad_to_rad

omega_ns = 13 * grad_to_rad  # скорость вращения НЗ - будет меняться только угол fi_mu!

# Parameters
MSun = config.MSun  # масса молнца г
G = config.G  # гравитационная постоянная см3·с−2·г−1
M_ns = config.M_ns  # масса нз г
R_ns = config.R_ns  # радиус нз см
sigmStfBolc = config.sigmStfBolc  # постоянная стефана больцмана в сгс

# new args (for new func)
dRe_div_Re = config.dRe_div_Re  # взял просто число
# M_accretion_rate = 10 ** 38 * R_ns / G / MSun  # темп аккреции
M_accretion_rate = config.M_accretion_rate
ksi_rad = config.ksi_rad
H = config.H  # магнитное поле стр 19 над формулой 37
a_portion = config.a_portion  # a - в азимутальном направлении поток занимает фиксированную долю a полного круга 2πR sinθ

# mu = 10 ** 30 * G  # магнитный момент см3
mu = config.mu  # магнитный момент Гаусс * см3
R_alfven = (mu ** 2 / (2 * M_accretion_rate * (2 * G * M_ns) ** (1 / 2))) ** (2 / 7)  # альфвеновский радиус
ksi_param = 0.5
R_e = ksi_param * R_alfven
print("R_e :%f" % R_e)

# количество шагов
N_fi_accretion = 100
N_theta_accretion = 100

theta_accretion_begin = np.arcsin((R_ns / R_e) ** (1 / 2))  # от поверхности NS


# формула 2 в статье
def delta_distance(theta):
    # R=R_e * sin_theta ** 2
    return R_e * np.sin(theta) ** 3 / (1 + 3 * np.cos(theta) ** 2) ** (1 / 2) * dRe_div_Re


# формула 3 в статье
def A_normal(a, theta):
    # a - в азимутальном направлении поток занимает фиксированную долю a полного круга 2πR sinθ
    return 2 * delta_distance(theta) * 2 * np.pi * a * R_e * np.sin(theta) ** 3


Teffbs, Teff, ksiShock = teffBsCheck.get_Teff_distribution(N_theta_accretion, M_accretion_rate, H, dRe_div_Re, R_e,
                                                           ksi_rad, delta_distance(theta_accretion_begin),
                                                           A_normal(a_portion, theta_accretion_begin))

ksiStop1 = 1.
ksiInc1 = - (ksiShock - ksiStop1) / N_theta_accretion
ksi1 = np.arange(ksiShock, ksiStop1, ksiInc1)
print("ksiShock = %f" % ksiShock)

# Plot results
fig = plt.figure(1, figsize=(8, 8))
ax3 = fig.add_subplot(111)

ax3.scatter(ksi1[::-1], np.log10(Teff), marker='*', alpha=0.5, color='b', label='Teff')
# ax3.scatter(ksi1[::-1], np.log10(Teffbs), color='r', label='Teffbs')

# ax3.plot(ksi1[::-1], np.log10(Teff), 'b', alpha=1, label='Teff')
ax3.plot(ksi1[::-1], np.log10(Teffbs), 'r', linestyle='-', label='Teffbs')  # аналитическое решение

ax3.set_xlabel('ksi')
ax3.set_ylabel('log10T')
ax3.legend()
plt.show()
