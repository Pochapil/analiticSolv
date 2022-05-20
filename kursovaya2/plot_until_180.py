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

number_of_files = 10
integral = [0] * number_of_files
for i in range(number_of_files):
    current_file_count = file_count + i * 3
    file_name = "save%d.txt" % current_file_count
    integral[i] = np.loadtxt(file_name)

fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot(111)
for i in range(number_of_files):
    current_file_count = file_count + i * 3
    ax3.plot(phi_for_plot, integral[i],
             label=r"$\omega=%d \, \mu=%d$" % ((current_file_count // 3) * 15, (current_file_count % 3) * 15))

ax3.set_xlabel('phase')
ax3.set_ylabel("L, erg/s")
ax3.legend()
plt.yscale('log')
plt.show()


fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot(111)
for i in range(number_of_files):
    current_file_count = file_count + i * 3
    ax3.plot(phi_for_plot, integral[i] / (max(integral[i])) * 2,
             label=r"$\omega=%d \, \mu=%d$" % ((current_file_count // 3) * 15, (current_file_count % 3) * 15))
ax3.plot(phi_for_plot, np.cos(phi_for_plot * 2 * np.pi) + 1, 'black', marker='*', label=r"$\cos{(phase)} + 1$")
ax3.set_xlabel('phase')
ax3.set_ylabel("$2 L/L_{max}$")
ax3.legend()
plt.show()
