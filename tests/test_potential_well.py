import numpy as np
import matplotlib.pyplot as plt

from pytrans.potential_well import PotentialWell
w = PotentialWell(130e-6, 0.1, 1e6, 1.3e6, 30)
# w = PotentialWell([0, 130e-6], [0.1, 0.2], 1e6, 1.3e6, 30)

print(w.x0.shape)
print(w.hessian.shape)
# # h, v = np.linalg.eig(w.hessian)

# print(curv_to_freq(h) * 1e-6)
# print(v)

x = np.arange(-600, 600, 2) * 1e-6
# x = x[w.roi(x)]
print(w.timesteps)
for j in range(w.timesteps):
    pot = w.potential(x, j)

    l, = plt.plot(x * 1e6, pot)
    plt.plot(x * 1e6, w.gaussian_potential(x, j), ls='--', color=l.get_color())
plt.show()
