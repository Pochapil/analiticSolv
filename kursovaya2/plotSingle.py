import numpy as np
import matplotlib.pyplot as plt
import matplotlib

file_count = 5

grad_to_rad = np.pi / 180
betta_rotate = (file_count // 3) * 15
betta_mu = (file_count % 3) * 15

print("betta_rotate = %d" % betta_rotate)
print("betta_mu = %d" % betta_mu)

phi_for_plot = np.loadtxt("phi_for_plot.txt")

integral5 = np.loadtxt("save5.txt")
integral11 = np.loadtxt("save11.txt")
integral17 = np.loadtxt("save17.txt")
integral29 = np.loadtxt("save29.txt")
integral41 = np.loadtxt("save41.txt")
integral53 = np.loadtxt("save53.txt")
integral65 = np.loadtxt("save65.txt")

file_count_set = [5, 11, 17, 29, 41, 53, 65]
integral_set = [integral5, integral11, integral17, integral29, integral41, integral53, integral65]

fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot(111)
for i in range(len(file_count_set)):
    ax3.plot(phi_for_plot, integral_set[i],
             label=r"$\omega=%d \, \mu=%d$" % ((file_count_set[i] // 3) * 15, (file_count_set[i] % 3) * 15))

ax3.set_xlabel('phase')
ax3.set_ylabel("L, erg/s")
ax3.legend()
plt.yscale('log')
plt.show()

i = 0
fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot(111)
ax3.plot(phi_for_plot, integral_set[i],
         label=r"$\omega=%d \, \mu=%d$" % ((file_count_set[i] // 3) * 15, (file_count_set[i] % 3) * 15))

ax3.set_xlabel('phase')
ax3.set_ylabel("L, erg/s")
ax3.legend()
plt.yscale('log')
plt.show()

fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot(111)
for i in range(len(file_count_set)):
    ax3.plot(phi_for_plot, integral_set[i] / (max(integral_set[i])) * 2,
             label=r"$\omega=%d \, \mu=%d$" % ((file_count_set[i] // 3) * 15, (file_count_set[i] % 3) * 15))
ax3.plot(phi_for_plot, np.cos(phi_for_plot * 2 * np.pi) + 1, 'black',marker='*', label=r"$\cos{(phase)} + 1$")
ax3.set_xlabel('phase')
ax3.set_ylabel("$2 L/L_{max}$")
ax3.legend()
plt.show()
