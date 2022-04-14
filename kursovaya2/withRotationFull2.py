import geomtetricTask.matrix as matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.integrate

import matplotlib.cm as cm
from matplotlib.colors import Normalize

matplotlib.use("TkAgg")
import newArgsFunc2 as newArgsFunc

import config  # const

# смотри RotationSingleColorBar.py

grad_to_rad = np.pi / 180
print(grad_to_rad)
# угол между нормалью к двойной системе и наблюдателем
i_angle = 30 * grad_to_rad
# вектор на наблюдателя в системе координат двойной системы
e_obs = np.array([0, np.sin(i_angle), np.cos(i_angle)])

# угол между осью вращения системы и собственным вращенеим НЗ
betta_rotate = 12 * grad_to_rad
fi_rotate = 13 * grad_to_rad

# угол между собственным вращенеим НЗ и магнитной осью
betta_mu = 15 * grad_to_rad
fi_mu_0 = 0 * grad_to_rad

# omega_ns = 4 * grad_to_rad  # скорость вращения НЗ - будет меняться только угол fi_mu!

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
print(M_accretion_rate)
ksi_rad = config.ksi_rad
H = config.H  # магнитное поле стр 19 над формулой 37
a_portion = config.a_portion  # a - в азимутальном направлении поток занимает фиксированную долю a полного круга 2πR sinθ

mu = config.mu  # магнитный момент Гаусс * см3
R_alfven = (mu ** 2 / (2 * M_accretion_rate * (2 * G * M_ns) ** (1 / 2))) ** (2 / 7)  # альфвеновский радиус
ksi_param = 0.5
R_e = ksi_param * R_alfven
print("R_e :%f" % R_e)
print("R_e/R_* = %f" % (R_e / R_ns))

# количество шагов
N_fi_accretion = 100
N_theta_accretion = 100

theta_accretion_begin = np.arcsin((R_ns / R_e) ** (1 / 2))  # от поверхности NS - угол при котором радиус = радиусу НЗ


# формула 2 в статье
def delta_distance(theta):
    # R=R_e * sin_theta ** 2
    return R_e * np.sin(theta) ** 3 / (1 + 3 * np.cos(theta) ** 2) ** (1 / 2) * dRe_div_Re


# формула 3 в статье
def A_normal(theta):
    # a - в азимутальном направлении поток занимает фиксированную долю a полного круга 2πR sinθ
    return 2 * delta_distance(theta) * 2 * np.pi * a_portion * R_e * np.sin(theta) ** 3


print("A|(R*)/R*2 = %f" % (A_normal(theta_accretion_begin) / R_ns ** 2))
print("delta(R*)/R* = %f" % (delta_distance(theta_accretion_begin) / R_ns))

# t_r = 1200 * a_portion/

Teff, ksiShock = newArgsFunc.get_Teff_distribution(N_theta_accretion, M_accretion_rate, R_e,
                                                   delta_distance(theta_accretion_begin),
                                                   A_normal(theta_accretion_begin))
# ksiShock = 4.523317
print("ksiShock :%f" % ksiShock)
print("Rshock/R*: %f" % ksiShock)
print("t2.3  %f" % (G * M_ns * M_accretion_rate / (R_ns * config.L_edd)))
# print("arcsin begin: %f" % (R_ns / R_e) ** (1 / 2))
# print("arcsin end: %f" % (R_ns * ksiShock / R_e) ** (1 / 2))

theta_accretion_end = np.arcsin((R_ns * ksiShock / R_e) ** (1 / 2))  # до шока - угол когда радиус = радиус шока

fi_accretion = 360 * grad_to_rad * 1.01  # полный круг для наложения. на карте были пробелы
print("theta_accretion_begin = %f" % (theta_accretion_begin / grad_to_rad))
print("theta_accretion_end = %f" % (theta_accretion_end / grad_to_rad))

# шаги по углам для интегрирования
step_fi_accretion = fi_accretion / N_fi_accretion
step_theta_accretion = (theta_accretion_end - theta_accretion_begin) / N_fi_accretion

# для отрисовки карты и интеграла
fi_range = np.array([step_fi_accretion * i for i in range(N_fi_accretion)])
theta_range = np.array([theta_accretion_begin + step_theta_accretion * j for j in range(N_theta_accretion)])
cos_psi_range = np.empty([N_fi_accretion, N_theta_accretion])

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

# цикл для поворотов, сколько точек на графике интегралов
# t_max = 40  # sec

omega_ns = 8  # скорость вращения НЗ - будет меняться только угол fi_mu!
t_max = (360 // omega_ns) + (
    1 if 360 % omega_ns > 0 else 0)  # цикл для поворотов, сколько точек на графике интегралов - для полного поворота
omega_ns = omega_ns * grad_to_rad


def calculate_integral_distribution(t_max, N_fi_accretion, N_theta_accretion):
    integral_max = 0
    # sum_intense изотропная светимость ( * 4 pi еще надо)
    sum_intense = [0] * t_max
    # для интеграла по simpson
    sum_simps_integrate = [0] * t_max
    simps_integrate_step = [0] * N_fi_accretion
    simps_cos = [0] * N_theta_accretion  # cos для интеграла по симпсону
    for i1 in range(t_max):
        # поворот
        fi_mu = fi_mu_0 + omega_ns * i1
        # расчет матрицы поворота в магнитную СК и вектора на наблюдателя
        A_matrix_analytic = matrix.newMatrixAnalytic(fi_rotate, betta_rotate, fi_mu, betta_mu)

        # print("analytic matrix:")
        # print(A_matrix_analytic)
        e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
        # print("e_obs_mu%d: (%f, %f, %f), angle phi = %f" % (
        # i1, e_obs_mu[0, 0], e_obs_mu[0, 1], e_obs_mu[0, 2], np.arctan(e_obs_mu[0, 1] / e_obs_mu[0, 0])/grad_to_rad))
        # print("e_obs_mu%d: (%f, %f, %f)" % (i1, np.take(e_obs_mu, 0), np.take(e_obs_mu, 1), np.take(e_obs_mu, 2)))

        # sum_intense изотропная светимость ( * 4 pi еще надо)
        for i in range(N_fi_accretion):
            for j in range(N_theta_accretion):
                # cos_psi_range[i][j] = np.dot(e_obs_mu, matrix.newE_n(fi_range[i], theta_range[j]))
                cos_psi_range[i][j] = np.dot(e_obs_mu, array_normal[i * N_theta_accretion + j])
                if cos_psi_range[i][j] > 0:
                    sum_intense[i1] += sigmStfBolc * Teff[j] ** 4 * cos_psi_range[i][j] * dS[j]
                    simps_cos[j] = cos_psi_range[i][j]
                    # * S=R**2 * step_fi_accretion * step_teta_accretion
                else:
                    simps_cos[j] = 0

            simps_integrate_step[i] = sigmStfBolc * scipy.integrate.simps(Teff ** 4 * simps_cos * dS_simps, theta_range)
        # находим позицию максимума
        if integral_max < sum_intense[i1]:
            position_of_max = i1
            integral_max = sum_intense[i1]

        sum_simps_integrate[i1] = scipy.integrate.simps(simps_integrate_step, fi_range)
    return sum_intense, sum_simps_integrate, position_of_max


sum_intense, sum_simps_integrate, position_of_max = calculate_integral_distribution(t_max, N_fi_accretion,
                                                                                    N_theta_accretion)
print("max: %d" % position_of_max)


def plot_map_cos(n_pos, position_of_max, t_max, N_fi_accretion, N_theta_accretion, row_number, column_number):
    number_of_plots = row_number * column_number

    crf = [0] * number_of_plots
    cr = [0] * number_of_plots

    fig, axes = plt.subplots(nrows=row_number, ncols=column_number, figsize=(8, 8), subplot_kw={'projection': 'polar'})
    # сдвигаем графики относительно позиции максимума. чтобы макс был на (0,0)
    row_figure = 0
    column_figure = 0
    # fi_mu = fi_mu_0
    for i1 in range(number_of_plots):
        fi_mu = fi_mu_0 + omega_ns * (n_pos + i1 + position_of_max)
        # расчет матрицы поворота в магнитную СК и вектора на наблюдателя
        A_matrix_analytic = matrix.newMatrixAnalytic(fi_rotate, betta_rotate, fi_mu, betta_mu)

        # A_matrix_analytic = matrix.newRy(betta_mu) @ matrix.newRz(fi_mu) @ matrix.newRy(betta_rotate) \
        #                 @ matrix.newRz(fi_rotate)

        # print("analytic matrix:")
        # print(A_matrix_analytic)
        count_0 = 0
        e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
        for i in range(N_fi_accretion):
            for j in range(N_theta_accretion):
                cos_psi_range[i][j] = np.dot(e_obs_mu, array_normal[i * N_fi_accretion + j])
                if cos_psi_range[i][j] < 0:
                    count_0 += 1

        crf[i1] = axes[row_figure, column_figure].contourf(fi_range, theta_range / grad_to_rad,
                                                           cos_psi_range.transpose(), vmin=-1, vmax=1)
        if count_0 > 0:
            cr[i1] = axes[row_figure, column_figure].contour(fi_range, theta_range / grad_to_rad,
                                                             cos_psi_range.transpose(),
                                                             [0.], colors='w')
        column_figure += 1
        if column_figure == column_number:
            column_figure = 0
            row_figure += 1

    cbar = fig.colorbar(crf[i1], ax=axes[:], shrink=0.8, location='right')

    plt.show()


def plot_map_cos_in_range(position_of_max, t_max, N_fi_accretion, N_theta_accretion, row_number, column_number):
    number_of_plots = row_number * column_number

    crf = [0] * number_of_plots
    cr = [0] * number_of_plots

    fig, axes = plt.subplots(nrows=row_number, ncols=column_number, figsize=(8, 8), subplot_kw={'projection': 'polar'})
    # сдвигаем графики относительно позиции максимума. чтобы макс был на (0,0)
    row_figure = 0
    column_figure = 0

    cmap = cm.get_cmap('viridis')
    normalizer = Normalize(-1, 1)
    im = cm.ScalarMappable(norm=normalizer)

    fi_mu_max = fi_mu_0 + omega_ns * position_of_max
    for i1 in range(number_of_plots):
        fi_mu = fi_mu_max + omega_ns * (t_max // (number_of_plots - 1)) * i1
        # расчет матрицы поворота в магнитную СК и вектора на наблюдателя
        A_matrix_analytic = matrix.newMatrixAnalytic(fi_rotate, betta_rotate, fi_mu, betta_mu)

        # A_matrix_analytic = matrix.newRy(betta_mu) @ matrix.newRz(fi_mu) @ matrix.newRy(betta_rotate) \
        #                 @ matrix.newRz(fi_rotate)

        # print("analytic matrix:")
        # print(A_matrix_analytic)
        count_0 = 0
        e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
        # e_obs_mu = np.array([0,1,-1])
        for i in range(N_fi_accretion):
            for j in range(N_theta_accretion):
                cos_psi_range[i][j] = np.dot(e_obs_mu, array_normal[i * N_fi_accretion + j])
                if cos_psi_range[i][j] < 0:
                    count_0 += 1

        crf[i1] = axes[row_figure, column_figure].contourf(fi_range, theta_range / grad_to_rad,
                                                           cos_psi_range.transpose(), vmin=-1, vmax=1, cmap=cmap,
                                                           norm=normalizer)
        if count_0 > 0:
            cr[i1] = axes[row_figure, column_figure].contour(fi_range, theta_range / grad_to_rad,
                                                             cos_psi_range.transpose(),
                                                             [0.], colors='w')

        axes[row_figure, column_figure].set_title(
            "phase = %.2f" % (omega_ns * (t_max // (number_of_plots - 1)) * i1 / (2 * np.pi)))
        column_figure += 1
        if column_figure == column_number:
            column_figure = 0
            row_figure += 1

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    cbar = fig.colorbar(im, ax=axes[:, :], shrink=0.7, location='right')
    plt.show()


for i in range(t_max):
    print("%d - " % i, end='')
    print(sum_intense[i], sum_simps_integrate[i])

fi_for_plot = list(omega_ns * i / (2 * np.pi) for i in range(t_max))
fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot(111)
# чтобы максимум был сначала - [position_of_max:], [0:position_of_max]
# ax3.plot(fi_for_plot, np.append(sum_intense[position_of_max:], sum_intense[0:position_of_max]), 'b', label='rectangle')
ax3.plot(fi_for_plot, np.append(sum_simps_integrate[position_of_max:], sum_simps_integrate[0:position_of_max]), 'r',
         label='simps')
ax3.set_xlabel('phase')
ax3.set_ylabel("isotropic luminosity, erg/s")
ax3.legend()
# ax3.yscale('log')
plt.yscale('log')

print(M_accretion_rate)
print(H)

n_pos = 10
row_number = 2
column_number = 3
# plot_map_cos(n_pos, position_of_max, t_max, N_fi_accretion, N_theta_accretion, row_number, column_number)
plot_map_cos_in_range(position_of_max, t_max, N_fi_accretion, N_theta_accretion, row_number, column_number)

plt.show()

# from mpl_toolkits.mplot3d import Axes3D
#
# ax = Axes3D(fig)
#
# x = y = z = np.zeros((N_fi_accretion, N_theta_accretion))
#
# for i in range(N_theta_accretion):
#     for j in range(N_fi_accretion):
#         x[i, j] = np.cos(fi_range[j])
#         y[i, j] = np.sin(fi_range[j])
#         z[i, j] = np.sin(theta_range[i]) ** 2  # bell curve
#
# x = x.flatten()
# y = y.flatten()
# z = z.flatten()
#
# ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0, antialiased=False)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

# theta_range = np.arange(0, 2*np.pi, 2*np.pi/N_theta_accretion)
theta_range = np.arange(0, np.pi/2, np.pi/2/N_theta_accretion)
fi_range = np.arange(0, np.pi/2, np.pi/2/N_fi_accretion)

r, p = np.meshgrid(np.sin(theta_range) ** 2, fi_range)
x = r * np.cos(p)
y = r * np.sin(p)
z = r * np.cos(theta_range)

# x = np.cos(fi_range)
# y = np.sin(fi_range)
# x, y = np.meshgrid(x, y)
# z, z1 = np.meshgrid(np.sin(theta_range) ** 2, fi_range)
#

# z = np.sin(theta_range) ** 2
# z, z1 = np.meshgrid(z, fi_range)


# from scipy.interpolate import griddata
#
# z = griddata((np.cos(fi_range),np.sin(fi_range)), z, (x, y), method='cubic')
# ax.plot3D(x, y, z, 'gray')


ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

plt.show()
