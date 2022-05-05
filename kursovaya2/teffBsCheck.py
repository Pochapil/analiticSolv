import numpy as np
from scipy.integrate import odeint
import scipy.special as special
import config
import matplotlib.pyplot as plt


def get_Teff_distribution(number_of_steps, M_accretion_rate, H, dRe_div_Re, R_e, ksi_rad, delta_ns, A_normal):
    # решение зависит от n размера пространства !!! взял n=3 везде
    # Parameters
    # const
    MSun = config.MSun  # масса молнца г
    G = config.G  # гравитационная постоянная см3·с−2·г−1
    c = config.c  # скорость света см/с
    sigmaT = config.sigmaT  # сечение томсона см-2
    massP = config.massP  # масса протона г
    # kT = 3.9832335 * 10 ** (-1)  # томсоновская непрозрачность
    kT = 0.4
    a_rad_const = config.a_rad_const  # радиационная константа p=aT**4  эрг см-3 К-4
    sigmStfBolc = config.sigmStfBolc  # постоянная стефана больцмана в сгс

    # neutron star
    # H = 10 ** 13  # магнитное поле стр 19 над формулой 37
    R_ns = config.R_ns  # радиус нз см
    M_ns = config.M_ns  # масса нз г
    l0 = 2 * 10 ** 5  # длина аккреции на поверхности м взял как в статье
    d0 = 5 * 10 ** 3  # ширина аккреции на поверхности м взял как в статье стр 4
    u0 = 3 * H ** 2 / 8 / np.pi  # значение плотности излучения на поверхности

    # formulas
    k = config.k  # opacity непрозрачность взял томсоновскую  стр 12 (под 11 формулой ?)
    s = c * R_ns / k / d0 ** 2  # поток массы при условии что gamma =1 (мб из 3 формулы ?) ур стр 14 ур 18 и стр 17 под ур 30
    # мб нужно умножить на 2 если 2 полюса
    # s = 2 * 10 ** 18 / (l0 * d0)  # поток массы M* взял 10**18 г/с эдингтоновский темп

    L2zv = 2 * l0 / d0 * c / k * G * M_ns  # предельная светимость L** стр 7 формула 3
    Lt = 2 * l0 * d0 * G * M_ns / R_ns * s  # светимость аккреции 13 стр формула 14
    # Lt = 2 * L2zv  # светимость аккреции
    # gamma = 1  # параметр отношение темпов аккреции
    # gamma = L2zv / Lt  # параметр отношение темпов аккреции под 30 формулой
    # eta = (8 / 21 * u0 * d0 ** 2 * k / (c * (2 * G * M * R) ** (1 / 2))) ** (1 / 4)  # константа
    # eta = 18  # взял из 19 стр над 37 формулой
    # eta = 16.7477  # взял из сообщения в телеге

    # gamma = 1  # параметр отношение темпов аккреции
    # eta = 16.7477  # взял из сообщения в телеге

    # 44 формула статья
    gamma = (c * R_ns * A_normal * 3) / (k * delta_ns ** 2 * M_accretion_rate * 2 * ksi_rad)
    # 45 формула статья
    eta = ((8 * k * u0 * delta_ns ** 2 * 2 * ksi_rad) / (21 * c * (2 * G * M_ns * R_ns) ** (1 / 2) * 3)) ** (1 / 4)

    # 30 формула, du/dksi; dv/dksi = производная от 3 равенства
    # возвращает u, v
    def func(y, ksi, params):
        u, v = y  # unpack current values of y
        gamma, s, G, M, R = params  # unpack parameters
        derivs = [3 * s * G * M / R * ksi ** (-5) / v,  # list of dy/dt=f functions
                  gamma * v - 3 * v / ksi - 9 / 4 * s * G * M / R * ksi ** (-5) / u]
        return derivs

    # 34 формула - из нее нахожу с помощью метода ньютона
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
        ksi1 = 4.3 * R_ns / R_ns
        ksi2 = nuton(ksi1)
        while np.abs((ksi1 - ksi2)) > delta:
            ksi1 = ksi2
            ksi2 = nuton(ksi1)
        return ksi2  # rs/R - находим радиус ударной волны

    ksiShock = findKsiShock()
    # 31 формула - значения функций в точке ksiShock - граничные значения для численных расчетов
    v1 = -1 / 7 * (2 * G * M_ns / R_ns) ** (1 / 2) * ksiShock ** (-1 / 2)
    u1 = -3 / 4 * s * (G * M_ns / R_ns) * ksiShock ** (-4) / v1  # зависит от n размера пространства !!! взял n=3 везде

    # Bundle parameters for ODE solver
    params = [gamma, s, G, M_ns, R_ns]
    # Bundle initial conditions for ODE solver
    y0 = [u1, v1]

    ksiStop1 = 1.
    ksiInc1 = - (ksiShock - ksiStop1) / number_of_steps
    ksi1 = np.arange(ksiShock, ksiStop1, ksiInc1)
    solution_before_ksi = odeint(func, y0, ksi1, args=(params,), mxstep=5000000)  # от 0 до ксишок

    # analytic solve bs

    # 35 формула
    betta = 1 - gamma * np.exp(gamma) * (special.expn(1, gamma) - special.expn(1, gamma * ksiShock))

    # 32 формула - аналитическое решение
    def u(ksi):
        return u0 * (1 - np.exp(gamma) / betta * (special.expn(2, gamma) - special.expn(2, gamma * ksi) / ksi)) ** 4

    def v(ksi):
        return (3 / 4 * s * G * M_ns / R_ns * np.exp(gamma * ksi) / (ksi ** 3) * (
                1 / ksi * special.expn(2, gamma * ksi) + betta * np.exp(-gamma) - special.expn(2, gamma))) / -u(ksi)

    # сливаю в 1 массив, объединяя интервалы
    u_numerical_solution = solution_before_ksi[::-1, 0]

    T = (u_numerical_solution / a_rad_const) ** (1 / 4)
    ksi_bs = ksi1[::-1]
    Tbs = (u(ksi_bs) / a_rad_const) ** (1 / 4)  # настоящее аналитическое решение

    e = c / (k * s * d0)  # формула 18 стр 14
    # e = gamma * d0 / R_ns # под 30 формулой
    # e = c / k / (M_accretion_rate / A_normal) / delta_ns

    # 21 стр конец 2 абзаца
    def fTheta():
        u = solution_before_ksi[::-1, 0]
        v = solution_before_ksi[::-1, 1]
        x = ksi1[::-1]
        return -2 / 3 * e * x ** (3 / 2) * u * v

    # 21 стр конец 2 абзаца
    def fThetabs(x):
        return -2 / 3 * e * x ** (3 / 2) * u(x) * v(x)

    # ro = 1  # плотность падающего газа
    # Fr(ksi) 30 формула 17 стр
    def fr(x):
        return 4 / 3 * u(x) * v(x) + s * G * M_ns / R_ns * x ** (-4)

    # 19 стр под конец
    def q(ksi):
        return (ksi ** 3 * fr(ksi) - ksiShock ** 3 * fr(ksiShock)) * R_ns / (s * G * M_ns)

    # 30 формула 3 уравнение
    def frCalc(u, v, x):
        return 4 / 3 * u * v + s * G * M_ns / R_ns * x ** (-4)

    def qCalc(u, v, ksi):
        return (ksi ** 3 * frCalc(u, v, ksi) - ksiShock ** 3 * frCalc(u, v, ksi)) * R_ns / (s * G * M_ns)

    # получаем эффективную температуру из закона Стефана-Больцмана
    Teff = (fTheta() / sigmStfBolc) ** (1 / 4)
    Teffbs = (fThetabs(ksi_bs) / sigmStfBolc) ** (1 / 4)

    fig = plt.figure(figsize=(8, 8))

    ax5 = fig.add_subplot(121)
    ax5.plot(ksi1[::-1], solution_before_ksi[::-1, 0], 'b', label='u')
    ax5.plot(ksi1[::-1], u(ksi1[::-1]), 'r', label='ubs')
    ax5.set_xlabel('ksi')
    ax5.set_ylabel('u')
    ax5.legend()

    ax6 = fig.add_subplot(122)
    ax6.plot(ksi1[::-1], solution_before_ksi[::-1, 1] / c, 'b', label='v/c')
    ax6.plot(ksi1[::-1], v(ksi1[::-1]) / c, 'r', label='vbs/c')
    ax6.set_xlabel('ksi')
    ax6.set_ylabel('v/c')
    ax6.legend()

    plt.show()

    # проверка на соответствие 5 рисунку в БС
    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(111)

    ax1.plot(ksi1[::-1], np.log10(Teff), marker='*', alpha=0.5, color='b', label='Teff')
    ax1.plot(ksi1[::-1], np.log10(T), linestyle='--', color='g', label='Tin')

    ax1.plot(ksi1[::-1], np.log10(Teffbs), alpha=0.5, color='r', label='TeffBS')
    ax1.plot(ksi1[::-1], np.log10(Tbs), linestyle='-.', color='k', label='TinBS')

    ax1.legend()

    plt.show()

    print("e = %.4f" % e)

    return Teffbs, Teff, ksiShock,
