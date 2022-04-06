import matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import finalFiles.p0126.finalWithFunc as finalWithFunc

# смотри polarPlot

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
MSun = 2 * 10 ** 33  # масса молнца г
G = 6.67 * 10 ** (-8)  # гравитационная постоянная см3·с−2·г−1
M_ns = 1.4 * MSun  # масса нз г
R_ns = 10 ** 6  # радиус нз см
sigmStfBolc = 5.67 * 10 ** (-5)  # постоянная стефана больцмана в сгс

M_accretion_rate = 10 ** 38 * R_ns / G / MSun  # темп аккреции
# mu = 10 ** 30 * G  # магнитный момент см3
mu = 10 ** 30  # магнитный момент Гаусс * см3
R_alfven = (mu ** 2 / (2 * M_accretion_rate * (G * M_ns) ** (1 / 2))) ** (2 / 7)  # альфвеновский радиус
ksi_param = 0.5
R_e = ksi_param * R_alfven
print("R_e :%f" % R_e)

# количество шагов
N_fi_accretion = 100
N_theta_accretion = 100

Teff, ksiShock = finalWithFunc.get_Teff_distribution(N_theta_accretion)
# ksiShock = 4.523317
print("ksiShock :%f" % ksiShock)

# print("arcsin begin: %f" % (R_ns / R_e) ** (1 / 2))
# print("arcsin end: %f" % (R_ns * ksiShock / R_e) ** (1 / 2))
theta_accretion_begin = np.arcsin((R_ns / R_e) ** (1 / 2))  # от поверхности NS
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

# для отрисовки карты и интеграла
fi_range = np.array([step_fi_accretion * i for i in range(N_fi_accretion)])
theta_range = np.array([theta_accretion_begin + step_theta_accretion * j for j in range(N_theta_accretion)])
e_n = np.empty([N_fi_accretion, N_theta_accretion])
cos_psi_range = np.empty([N_fi_accretion, N_theta_accretion])

# цикл для поворотов
t_max = 6  # 20 sec

row_number = 2
column_number = t_max // row_number

sum_intense = [0] * t_max
crf = [0] * t_max
cr = [0] * t_max

fig = plt.figure(figsize=(8, 8))
# fig, axs = plt.subplots(4, 5, figsize=(15, 6))
# axs = axs.ravel()
array_normal = []  # матрица нормалей чтобы не пересчитывать в циклах
# matrix_normal = np.empty([N_fi_accretion, N_theta_accretion])
for i in range(N_fi_accretion):
    for j in range(N_theta_accretion):
        # matrix_normal[i, j] = matrix.newE_n(fi_range[i], theta_range[j])
        array_normal.append(matrix.newE_n(fi_range[i], theta_range[j]))

for i1 in range(t_max):
    # поворот
    fi_mu = fi_mu + omega_ns * i1
    # расчет матрицы поворота в магнитную СК и вектора на наблюдателя
    A_matrix_analytic = matrix.newMatrixAnalytic(fi_rotate, betta_rotate, fi_mu, betta_mu)
    # print("analytic matrix:")
    # print(A_matrix_analytic)

    e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
    print("e_obs_mu%d: (%f, %f, %f)" % (i1, e_obs_mu[0, 0], e_obs_mu[0, 1], e_obs_mu[0, 2]))
    # print("e_obs_mu%d: (%f, %f, %f)" % (i1, np.take(e_obs_mu, 0), np.take(e_obs_mu, 1), np.take(e_obs_mu, 2)))

    # изотропная светимость ( * 4 pi еще надо)
    for i in range(N_fi_accretion):
        for j in range(N_theta_accretion):
            dl = R_e * (3 * np.cos(theta_range[j]) ** 2 + 1) ** (1 / 2) * np.sin(theta_range[j]) * step_theta_accretion
            dfi = R_e * np.sin(theta_range[j]) ** 3 * step_fi_accretion  # R=R_e * sin_theta ** 2; R_fi = R * sin_theta
            dS = dfi * dl  # единичная площадка при интегрировании
            # cos_psi_range[i][j] = np.dot(e_obs_mu, matrix.newE_n(fi_range[i], theta_range[j]))
            cos_psi_range[i][j] = np.dot(e_obs_mu, array_normal[i * N_fi_accretion + j])
            if cos_psi_range[i][j] > 0:
                sum_intense[i1] += sigmStfBolc * Teff[j] ** 4 * cos_psi_range[i][j] * dS
                # * S=R**2 * step_fi_accretion * step_teta_accretion

    extent = np.min(theta_range) / grad_to_rad, np.max(theta_range) / grad_to_rad, np.min(
        fi_range) / grad_to_rad, np.max(
        fi_range) / grad_to_rad

    # fig = plt.figure(figsize=(5, 5))
    axs = fig.add_subplot(row_number, column_number, i1 + 1, projection='polar')
    crf[i1] = axs.contourf(fi_range, theta_range / grad_to_rad, cos_psi_range.transpose())
    cr[i1] = axs.contour(fi_range, theta_range / grad_to_rad, cos_psi_range.transpose(), [0.], colors='w')
    # ax4 = fig.add_subplot(111, projection='polar')
    # сама карта
    # c1 = ax4.contourf(fi_range, theta_range / grad_to_rad, cos_psi_range.transpose())
    # выделяю контур 0
    # contour = ax4.contour(fi_range, theta_range / grad_to_rad, cos_psi_range.transpose(), [0.], colors='w')
    plt.colorbar(crf[i1])



for i in range(t_max):
    print("%d - " % i, end='')
    print(sum_intense[i])

fi_for_plot = list(omega_ns * i for i in range(t_max))
fig = plt.figure(figsize=(8, 8))
plt.plot(fi_for_plot, sum_intense)
plt.yscale('log')
plt.show()
