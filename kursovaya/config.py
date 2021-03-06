# Parameters

#const
MSun = 2 * 10 ** 33  # масса молнца г
G = 6.67 * 10 ** (-8)  # гравитационная постоянная см3·с−2·г−1
c = 3 * 10 ** 10  # скорость света см/с
sigmStfBolc = 5.67 * 10 ** (-5)  # постоянная стефана больцмана в сгс
a_rad_const = 7.5657 * 10 ** (-15)  # радиационная константа p=aT**4  эрг см-3 К-4
sigmaT = 6.652 * 10 ** (-25)  # сечение томсона см-2
massP = 1.67 * 10 ** (-24)  # масса протона г

#n_s const
M_ns = 1.4 * MSun  # масса нз г
R_ns = 10 ** 6  # радиус нз см
H = 2 * 10 ** 13  # магнитное поле стр 19 над формулой 37
mu = 1 * 10 ** 31  # магнитный момент Гаусс * см3

# new args (for new func)
dRe_div_Re = 0.25  # взял просто число
M_accretion_rate = 10 ** 38 * R_ns / G / MSun  # темп аккреции
ksi_rad = 3 / 2
a_portion = 1  # a - в азимутальном направлении поток занимает фиксированную долю a полного круга 2πR sinθ
k = 0.35  # opacity непрозрачность
