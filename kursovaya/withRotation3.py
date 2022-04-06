import geomtetricTask.matrix as matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.integrate

matplotlib.use("TkAgg")
import newArgsFunc2 as newArgsFunc

import config  # const

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
M_accretion_rate = 10 ** 38 * R_ns / G / MSun  # темп аккреции
print(M_accretion_rate)
ksi_rad = config.ksi_rad
H = config.H  # магнитное поле стр 19 над формулой 37
a_portion = config.a_portion  # a - в азимутальном направлении поток занимает фиксированную долю a полного круга 2πR sinθ

# mu = 10 ** 30 * G  # магнитный момент см3
mu = config.mu  # магнитный момент Гаусс * см3
R_alfven = (mu ** 2 / (2 * M_accretion_rate * (G * M_ns) ** (1 / 2))) ** (2 / 7)  # альфвеновский радиус
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
def A_normal(theta):
    # a - в азимутальном направлении поток занимает фиксированную долю a полного круга 2πR sinθ
    return 2 * delta_distance(theta) * 2 * np.pi * a_portion * R_e * np.sin(theta) ** 3


Teff, ksiShock = newArgsFunc.get_Teff_distribution(N_theta_accretion, M_accretion_rate, R_e,
                                                   delta_distance(theta_accretion_begin),
                                                   A_normal(theta_accretion_begin))
# ksiShock = 4.523317
print("ksiShock :%f" % ksiShock)

# print("arcsin begin: %f" % (R_ns / R_e) ** (1 / 2))
# print("arcsin end: %f" % (R_ns * ksiShock / R_e) ** (1 / 2))

theta_accretion_end = np.arcsin((R_ns * ksiShock / R_e) ** (1 / 2))  # до шока

fi_accretion = 360 * grad_to_rad * 1.01
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

# цикл для поворотов, сколько точек на графике интегралов
t_max = 6  # sec

row_number = 2
column_number = t_max // row_number

# sum_intense изотропная светимость ( * 4 pi еще надо)
sum_intense = [0] * t_max
crf = [0] * t_max
cr = [0] * t_max
# для интеграла по simpson
sum_simps_integrate = [0] * t_max
simps_integrate_step = [0] * N_fi_accretion
simps_cos = [0] * N_theta_accretion  # cos для интеграла по симпсону

array_normal = []  # матрица нормалей чтобы не пересчитывать в циклах
# matrix_normal = np.empty([N_fi_accretion, N_theta_accretion])
for i in range(N_fi_accretion):
    for j in range(N_theta_accretion):
        # matrix_normal[i, j] = matrix.newE_n(fi_range[i], theta_range[j])
        array_normal.append(matrix.newE_n(fi_range[i], theta_range[j]))

dS = []  # массив единичных площадок при интегрировании так как зависит только от theta посчитаю 1 раз
dS_simps = []
# формула 5 из статьи для dl
for j in range(N_theta_accretion):
    dl = R_e * (3 * np.cos(theta_range[j]) ** 2 + 1) ** (1 / 2) * np.sin(theta_range[j]) * step_theta_accretion
    dl_simps = R_e * (3 * np.cos(theta_range[j]) ** 2 + 1) ** (1 / 2) * np.sin(theta_range[j])
    dfi = R_e * np.sin(theta_range[j]) ** 3 * step_fi_accretion  # R=R_e * sin_theta ** 2; R_fi = R * sin_theta
    dfi_simps = R_e * np.sin(theta_range[j]) ** 3
    dS.append(dfi * dl)  # единичная площадка при интегрировании
    dS_simps.append(dfi_simps * dl_simps)

column_figure = 0
row_figure = 0
fig, axes = plt.subplots(nrows=row_number, ncols=column_number, figsize=(8, 8), subplot_kw={'projection': 'polar'})

for i1 in range(t_max):
    # поворот
    fi_mu = fi_mu + omega_ns
    # расчет матрицы поворота в магнитную СК и вектора на наблюдателя
    A_matrix_analytic = matrix.newMatrixAnalytic(fi_rotate, betta_rotate, fi_mu, betta_mu)
    # print("analytic matrix:")
    # print(A_matrix_analytic)
    e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
    print("e_obs_mu%d: (%f, %f, %f)" % (i1, e_obs_mu[0, 0], e_obs_mu[0, 1], e_obs_mu[0, 2]))
    # print("e_obs_mu%d: (%f, %f, %f)" % (i1, np.take(e_obs_mu, 0), np.take(e_obs_mu, 1), np.take(e_obs_mu, 2)))

    # sum_intense изотропная светимость ( * 4 pi еще надо)
    for i in range(N_fi_accretion):
        for j in range(N_theta_accretion):

            # cos_psi_range[i][j] = np.dot(e_obs_mu, matrix.newE_n(fi_range[i], theta_range[j]))
            cos_psi_range[i][j] = np.dot(e_obs_mu, array_normal[i * N_fi_accretion + j])
            if cos_psi_range[i][j] > 0:
                sum_intense[i1] += sigmStfBolc * Teff[j] ** 4 * cos_psi_range[i][j] * dS[j]
                simps_cos[j] = cos_psi_range[i][j]
                # * S=R**2 * step_fi_accretion * step_teta_accretion
            else:
                simps_cos[j] = 0

        simps_integrate_step[i] = sigmStfBolc * scipy.integrate.simps(Teff ** 4 * simps_cos * dS_simps, theta_range)

    sum_simps_integrate[i1] = scipy.integrate.simps(simps_integrate_step, fi_range)

    crf[i1] = axes[row_figure, column_figure].contourf(fi_range, theta_range / grad_to_rad, cos_psi_range.transpose(),
                                                       vmin=-1, vmax=1)
    cr[i1] = axes[row_figure, column_figure].contour(fi_range, theta_range / grad_to_rad, cos_psi_range.transpose(),
                                                     [0.], colors='w')
    column_figure += 1
    if column_figure == column_number:
        column_figure = 0
        row_figure += 1

cbar = fig.colorbar(crf[t_max - 1], ax=axes[:], shrink=0.8, location='right')

for i in range(t_max):
    print("%d - " % i, end='')
    print(sum_intense[i], sum_simps_integrate[i])

fi_for_plot = list(omega_ns * i for i in range(t_max))
fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot(111)
ax3.plot(fi_for_plot, sum_intense, 'b', label='rectangle')
ax3.plot(fi_for_plot, sum_simps_integrate, 'r', label='simps')
ax3.set_xlabel('fi_rotate')
ax3.set_ylabel("integral")
ax3.legend()
# ax3.yscale('log')
plt.yscale('log')
plt.show()
