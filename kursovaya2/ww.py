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
    # находим позицию максимума
    if integral_max <= sum_intense[i1]:
        position_of_max = i1
        integral_max = sum_intense[i1]

    sum_simps_integrate[i1] = scipy.integrate.simps(simps_integrate_step, fi_range)

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
    # находим позицию максимума
    if integral_max <= sum_intense[i1]:
        position_of_max = i1
        integral_max = sum_intense[i1]

    sum_simps_integrate[i1] = scipy.integrate.simps(simps_integrate_step, fi_range)

print("max: %d" % position_of_max)
fig, axes = plt.subplots(nrows=row_number, ncols=column_number, figsize=(8, 8), subplot_kw={'projection': 'polar'})
# сдвигаем графики относительно позиции максимума. чтобы макс был на (0,0)
row_figure = (t_max - position_of_max) // column_number
column_figure = (t_max - position_of_max) % column_number
fi_mu = fi_mu_0
for i1 in range(t_max):
    fi_mu = fi_mu + omega_ns * i1
    # расчет матрицы поворота в магнитную СК и вектора на наблюдателя
    A_matrix_analytic = matrix.newMatrixAnalytic(fi_rotate, betta_rotate, fi_mu, betta_mu)
    # print("analytic matrix:")
    # print(A_matrix_analytic)
    e_obs_mu = np.dot(A_matrix_analytic, e_obs)  # переход в магнитную СК
    for i in range(N_fi_accretion):
        for j in range(N_theta_accretion):
            cos_psi_range[i][j] = np.dot(e_obs_mu, array_normal[i * N_fi_accretion + j])

    crf[i1] = axes[row_figure, column_figure].contourf(fi_range, theta_range / grad_to_rad, cos_psi_range.transpose(),
                                                       vmin=-1, vmax=1)
    cr[i1] = axes[row_figure, column_figure].contour(fi_range, theta_range / grad_to_rad, cos_psi_range.transpose(),
                                                     [0.], colors='w')
    column_figure += 1
    if column_figure == column_number:
        column_figure = 0
        row_figure += 1

    if (row_figure * column_number + column_figure > t_max - 1):
        column_figure = 0
        row_figure = 0

cbar = fig.colorbar(crf[position_of_max], ax=axes[:], shrink=0.8, location='right')
