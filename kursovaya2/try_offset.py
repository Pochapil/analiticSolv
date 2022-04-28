import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, \
    MaxNLocator, DictFormatter
import numpy as np
import matplotlib.pyplot as plt

# generate 100 random data points
# order the theta coordinates

# theta between 0 and 2*pi
theta = np.random.rand(100) * 2. * np.pi
theta = np.sort(theta)

# "radius" between 0 and a max value of 40,000
# as roughly in your example
# normalize the r coordinates and offset by 1 (will be clear later)
MAX_R = 40000.
radius = np.random.rand(100) * MAX_R
radius = radius / np.max(radius) + 1.

# initialize figure:
fig = plt.figure()

# set up polar axis
tr = PolarAxes.PolarTransform()

# define angle ticks around the circumference:
angle_ticks = [(0, r"$0$"),
               (.25 * np.pi, r"$\frac{1}{4}\pi$"),
               (.5 * np.pi, r"$\frac{1}{2}\pi$"),
               (.75 * np.pi, r"$\frac{3}{4}\pi$"),
               (1. * np.pi, r"$\pi$"),
               (1.25 * np.pi, r"$\frac{5}{4}\pi$"),
               (1.5 * np.pi, r"$\frac{3}{2}\pi$"),
               (1.75 * np.pi, r"$\frac{7}{4}\pi$")]

# set up ticks and spacing around the circle
grid_locator1 = FixedLocator([v for v, s in angle_ticks])
tick_formatter1 = DictFormatter(dict(angle_ticks))

# set up grid spacing along the 'radius'
radius_ticks = [(1., '0.0'),
                (1.5, '%i' % (MAX_R / 2.)),
                (2.0, '%i' % (MAX_R))]

grid_locator2 = FixedLocator([v for v, s in radius_ticks])
tick_formatter2 = DictFormatter(dict(radius_ticks))

# set up axis:
# tr: the polar axis setup
# extremes: theta max, theta min, r max, r min
# the grid for the theta axis
# the grid for the r axis
# the tick formatting for the theta axis
# the tick formatting for the r axis
grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                                  extremes=(2. * np.pi, 0, 2, 1),
                                                  grid_locator1=grid_locator1,
                                                  grid_locator2=grid_locator2,
                                                  tick_formatter1=tick_formatter1,
                                                  tick_formatter2=tick_formatter2)

ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
fig.add_subplot(ax1)

# create a parasite axes whose transData in RA, cz
aux_ax = ax1.get_aux_axes(tr)

aux_ax.patch = ax1.patch  # for aux_ax to have a clip path as in ax
ax1.patch.zorder = 0.9  # but this has a side effect that the patch is
# drawn twice, and possibly over some other
# artists. So, we decrease the zorder a bit to
# prevent this.

# plot your data:
aux_ax.plot(theta, radius)
plt.show()

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import geomtetricTask.matrix as matrix

grad_to_rad = np.pi / 180
phi_mu_0 = 12 * grad_to_rad
omega_ns = 11 * grad_to_rad
phi_rotate = 10 * grad_to_rad
betta_rotate = 10 * grad_to_rad
betta_mu = 12 * grad_to_rad
i_angle = 30 * grad_to_rad

e_obs = np.array([0, np.sin(i_angle), np.cos(i_angle)])

N_phi_accretion = 100
N_theta_accretion = 100

theta_accretion_begin = 11 * grad_to_rad
theta_accretion_end = 30 * grad_to_rad  # до шока - угол когда радиус = радиус шока
phi_accretion = 360 * grad_to_rad * 1.01

step_phi_accretion = phi_accretion / N_phi_accretion
step_theta_accretion = (theta_accretion_end - theta_accretion_begin) / N_phi_accretion

phi_range = np.array([step_phi_accretion * i for i in range(N_phi_accretion)])
theta_range = np.array([theta_accretion_begin + step_theta_accretion * j for j in range(N_theta_accretion)])
cos_psi_range = np.empty([N_phi_accretion, N_theta_accretion])

array_normal = []
for i in range(N_phi_accretion):
    for j in range(N_theta_accretion):
        # matrix_normal[i, j] = matrix.newE_n(phi_range[i], theta_range[j])
        array_normal.append(matrix.newE_n(phi_range[i], theta_range[j]))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw={'projection': 'polar'})

# для единого колорбара - взял из инета
cmap = cm.get_cmap('viridis')
normalizer = Normalize(-1, 1)
im = cm.ScalarMappable(norm=normalizer)

phi_mu_max = phi_mu_0 + omega_ns * 0

# сдвигаем графики относительно позиции максимума. чтобы макс был на (0,0)
phi_mu = phi_mu_max + omega_ns
# расчет матрицы поворота в магнитную СК и вектора на наблюдателя
A_matrix_analytic = matrix.newMatrixAnalytic(phi_rotate, betta_rotate, phi_mu, betta_mu)

count_0 = 0  # счетчик для контура 0 на карте
e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
# e_obs_mu = np.array([0,1,-1])
for i in range(N_phi_accretion):
    for j in range(N_theta_accretion):
        cos_psi_range[i][j] = np.dot(e_obs_mu, array_normal[i * N_phi_accretion + j])
        if cos_psi_range[i][j] < 0:
            count_0 += 1

crf = axes.contourf(phi_range, theta_range / grad_to_rad,
                          cos_psi_range.transpose(), vmin=-1, vmax=1, cmap=cmap,
                          norm=normalizer)
# попытки для сдвига 0 на картах
# axes[row_figure, column_figure].set_ylim(theta_range[0]/grad_to_rad, theta_range[N_theta_accretion-1]/grad_to_rad)
# axes[row_figure, column_figure].set_rorigin(-theta_accretion_end / grad_to_rad)
# axes[row_figure, column_figure].set_theta_zero_location('W', offset=50)
if count_0 > 0:
    cr = axes.contour(phi_range, theta_range / grad_to_rad,
                            cos_psi_range.transpose(),
                            [0.], colors='w')

axes.set_title("phase = %.2f" % (omega_ns / (2 * np.pi)))

axes.set_rorigin(-theta_accretion_begin / grad_to_rad)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
cbar = fig.colorbar(im, ax=axes, shrink=0.7, location='right')
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw={'projection': 'polar'})

# для единого колорбара - взял из инета
cmap = cm.get_cmap('viridis')
normalizer = Normalize(-1, 1)
im = cm.ScalarMappable(norm=normalizer)

phi_mu_max = phi_mu_0 + omega_ns * 0

# сдвигаем графики относительно позиции максимума. чтобы макс был на (0,0)
phi_mu = phi_mu_max + omega_ns
# расчет матрицы поворота в магнитную СК и вектора на наблюдателя
A_matrix_analytic = matrix.newMatrixAnalytic(phi_rotate, betta_rotate, phi_mu, betta_mu)

count_0 = 0  # счетчик для контура 0 на карте
e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
# e_obs_mu = np.array([0,1,-1])
for i in range(N_phi_accretion):
    for j in range(N_theta_accretion):
        cos_psi_range[i][j] = np.dot(e_obs_mu, array_normal[i * N_phi_accretion + j])
        if cos_psi_range[i][j] < 0:
            count_0 += 1

crf = axes.contourf(phi_range, theta_range / grad_to_rad,
                          cos_psi_range.transpose(), vmin=-1, vmax=1, cmap=cmap,
                          norm=normalizer)
# попытки для сдвига 0 на картах
# axes[row_figure, column_figure].set_ylim(theta_range[0]/grad_to_rad, theta_range[N_theta_accretion-1]/grad_to_rad)
# axes[row_figure, column_figure].set_rorigin(-theta_accretion_end / grad_to_rad)
# axes[row_figure, column_figure].set_theta_zero_location('W', offset=50)
if count_0 > 0:
    cr = axes.contour(phi_range, theta_range / grad_to_rad,
                            cos_psi_range.transpose(),
                            [0.], colors='w')

axes.set_title("phase = %.2f" % (omega_ns / (2 * np.pi)))

plt.subplots_adjust(hspace=0.5, wspace=0.5)
cbar = fig.colorbar(im, ax=axes, shrink=0.7, location='right')
plt.show()
