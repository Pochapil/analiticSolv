import numpy as np

# Parameters
s = 1  # хз что это
MSun = 2 * 10 ** 30  # масса молнца кг
G = 6.67 * 10 ** (-11)  # гравитационная постоянная
M = 1.5 * MSun  # масса нз кг
R = 10 ** 4  # радиус нз м

c = 3 * 10 ** 8  # скорость света м/с
l0 = 2 * 10 ** 3  # длина аккреции на поверхности м  взял как в статье
d0 = 5 * 10  # ширина аккреции на поверхности м   взял как в статье стр 4
k = 3.977 * 10 ** (-2)  # opacity непрозрачность взял томсоновскую  стр 12
L2zv = 2 * l0 / d0 * c / k * G * M  # предельная светимость L**
Lt = 2 * L2zv  # светимость аккреции

gamma = L2zv / Lt  #


ksiStop = 10
ksiInc = 0.05
ksi = np.arange(1., ksiStop, ksiInc)

# Call the ODE solver
y = np.ones((2, ksi.shape[0]))

def f(y, ksi):
    u = y[0]
    v = y[1]
    return [3 * s * G * M / R * ksi ** (-5) / v,
                      gamma * v - 3 * v / ksi - 9 / 4 * s * G * M / R * ksi ** (-5) / u]



g = f (y,ksi)
print(g)