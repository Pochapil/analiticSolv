import matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as Tk

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import finalFiles.p0126.finalWithFunc as finalWithFunc

grad_to_rad = np.pi / 180

# угол между нормалью к двойной системе и наблюдателем
i_angle = 30 * grad_to_rad
e_obs = np.array([0,  np.sin(i_angle), np.cos(i_angle)])

# угол между осью вращения системы и собственным вращенеим НЗ
betta_rotate = 0 * grad_to_rad
fi_rotate = 12 * grad_to_rad

# угол между собственным вращенеим НЗ и магнитной осью
betta_mu = 15 * grad_to_rad
fi_mu = 32 * grad_to_rad

# A_matrix_analitic = np.dot(matrix.newRy(betta_mu), matrix.newRz(fi_mu))
# A_matrix_analitic = np.dot(A_matrix_analitic, matrix.newRy(betta_rotate))
# A_matrix_analitic = np.dot(A_matrix_analitic, matrix.newRz(fi_rotate))

A_matrix_calc = matrix.newRy(betta_mu) @ matrix.newRz(fi_mu) @ matrix.newRy(betta_rotate) \
                @ matrix.newRz(fi_rotate)
print("calculated matrix:")
print(A_matrix_calc)

A_matrix_analytic = matrix.newMatrixAnalytic(fi_rotate, betta_rotate, fi_mu, betta_mu)
print("analytic matrix:")
print(A_matrix_analytic)

# углы в сферической СК (уже в переведенных) - в магнитной
theta_sphere = 43 * grad_to_rad
fi_sphere = 43 * grad_to_rad

# базисы в сферической СК
e_r = np.array(
    [np.sin(theta_sphere) * np.cos(fi_sphere), np.sin(theta_sphere) * np.sin(fi_sphere), np.cos(theta_sphere)])
e_theta = np.array(
    [np.cos(theta_sphere) * np.cos(fi_sphere), np.cos(theta_sphere) * np.sin(fi_sphere), -np.sin(theta_sphere)])
e_fi = np.array([-np.sin(fi_sphere), np.cos(fi_sphere), 0])

# единичный вектор вдоль силовых линий
e_l = (2 * np.cos(theta_sphere) * e_r + np.sin(theta_sphere) * e_theta) / (
        (3 * np.cos(theta_sphere) ** 2 + 1) ** (1 / 2))
# нормаль к силовым линиям
e_n = np.cross(e_l, e_fi)
print("e_n: ")
print(e_n)


# print("e_obs: (%f, %f, %f)" % (np.take(e_obs, 0), np.take(e_obs, 1), np.take(e_obs, 2)))

e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
cos_psi = np.dot(e_obs_mu, e_n)
print("cos_psi:")
print(cos_psi)

e_n = matrix.newE_n(fi_sphere, theta_sphere)
cos_psi = np.dot(e_obs_mu, e_n)
print("cos_psi analytic:")
print(cos_psi)

# Parameters

MSun = 2 * 10 ** 33  # масса молнца г
G = 6.67 * 10 ** (-8)  # гравитационная постоянная см3·с−2·г−1
M_ns = 1.4 * MSun  # масса нз г
R_ns = 10 ** 6  # радиус нз см

M_accretion_rate = 10 ** 38 * R_ns / G / MSun  # темп аккреции
# mu = 10 ** 30 * G  # магнитный момент см3
mu = 10 ** 30
R_alfven = (mu ** 2 / (2 * M_accretion_rate * (G * M_ns) ** (1 / 2))) ** (2 / 7)  # альфвеновский радиус
ksi_param = 0.5
R_e = ksi_param * R_alfven
print("R_e :%f" % R_e)

# количество шагов
N_fi_accretion = 100
N_theta_accretion = 100

Teff,ksiShock = finalWithFunc.get_Teff_distribution(N_theta_accretion)
#ksiShock = 4.523317
print("ksiShock :%f" % ksiShock)

# print("arcsin begin: %f" % (R_ns / R_e) ** (1 / 2))
# print("arcsin end: %f" % (R_ns * ksiShock / R_e) ** (1 / 2))
theta_accretion_begin = np.arcsin((R_ns / R_e) ** (1 / 2))  # от поверхности
theta_accretion_end = np.arcsin((R_ns * ksiShock / R_e) ** (1 / 2))  # до шока

# e_obs_mu = np.array([0, np.sin(30 * grad_to_rad), np.cos(30 * grad_to_rad)])  # взял по оси магнитной
# e_obs_mu = np.array([0, 0, 1])  # взял по оси магнитной
# theta_accretion_end = 90 * grad_to_rad

fi_accretion = 360 * grad_to_rad
print("theta_accretion_begin = %f" % (theta_accretion_begin / grad_to_rad))
print("theta_accretion_end = %f" % (theta_accretion_end / grad_to_rad))

# шаги по углам для интегрирования
step_fi_accretion = fi_accretion / N_fi_accretion
step_theta_accretion = (theta_accretion_end - theta_accretion_begin) / N_fi_accretion

sum_intense = 0
sigmStfBolc = 5.67 * 10 ** (-5)  # постоянная стефана больцмана в сгс


# для отрисовки карты
fi_range = np.array([step_fi_accretion * i for i in range(N_fi_accretion)])
theta_range = np.array([theta_accretion_begin + step_theta_accretion * j for j in range(N_theta_accretion)])
e_n = np.empty([N_fi_accretion, N_theta_accretion])
cos_psi_range = np.empty([N_fi_accretion, N_theta_accretion])

for i in range(N_fi_accretion):
    for j in range(N_theta_accretion):
        dl = R_e * (3 * np.cos(theta_range[j]) ** 2 + 1) ** (1 / 2) * np.sin(theta_range[j]) * step_theta_accretion
        dfi = R_e * np.sin(theta_range[j]) * step_fi_accretion
        dS = dfi * dl  # единичная площадка при интегрировании
        cos_psi_range[i][j] = np.dot(e_obs_mu, matrix.newE_n(fi_range[i], theta_range[j]))
        if cos_psi_range[i][j] > 0:
            sum_intense += sigmStfBolc * Teff[j] ** 4 * cos_psi_range[i][j] * dS
            # * S=R**2 * step_fi_accretion * step_teta_accretion

print(sum_intense)

print("e_obs_mu: (%f, %f, %f)" % (np.take(e_obs_mu, 0), np.take(e_obs_mu, 1), np.take(e_obs_mu, 2)))

extent = np.min(theta_range) / grad_to_rad, np.max(theta_range) / grad_to_rad, np.min(fi_range) / grad_to_rad, np.max(
    fi_range) / grad_to_rad

# azimuths = fi_range
# zeniths = theta_range

fig = plt.figure(figsize=(5, 5))
# fig.clf()
# ax3 = fig.add_subplot(121)
# c = plt.imshow(cos_psi_range, extent=extent)
# plt.colorbar(c)
# plt.title(' heat map of cos_psi ', fontweight="bold")
# ax3.set_xlabel('theta')
# ax3.set_ylabel('fi')
ax4 = fig.add_subplot(111, projection='polar')
c1 = ax4.contourf(fi_range, theta_range/grad_to_rad, cos_psi_range.transpose())

#contour = plt.contour(fi_range, theta_range/grad_to_rad, cos_psi_range.transpose(), colors='r')
contour = plt.contour(fi_range, theta_range/grad_to_rad, cos_psi_range.transpose(), [0.], colors='w')
plt.colorbar(c1)

plt.show()

# r_plot, theta_plot = np.meshgrid(zeniths, azimuths)
# fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
# c = ax.contourf(theta_plot, r_plot, cos_psi_range)
# plt.colorbar(c)
#
# plt.show()
