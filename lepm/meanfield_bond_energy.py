import numpy as np
import matplotlib.pyplot as plt

k = 1.0
eps1 = 1.0
eps2 = 1.0

step = 0.0001
tt = np.arange(0.0, 2.*np.pi + step, step)
delta = np.pi * 1.


integral = 1./float(len(tt)) * np.sum(np.sin(tt) * np.sin(tt + delta))
Ebond = 0.25 * k * (eps1**2 + eps2**2) - k * eps1 * eps2 * integral

print 'integral = ', integral
print '4k * Ebond = ', 4*k*Ebond

deltas = np.pi * np.arange(0, 2., 0.01)
integrals = np.zeros_like(deltas)
ii = 0
for delta in deltas:
    integrals[ii] = 1./float(len(tt)) * np.sum(np.sin(tt) * np.sin(tt + delta))
    ii += 1
Ebonds = 0.25 * k * (eps1**2 + eps2**2) - k * eps1 * eps2 * integrals

plt.plot(deltas/np.pi, integrals)
plt.plot(deltas/np.pi, 0.5*np.cos(deltas), 'r--')
plt.xlabel(r'$\delta/\pi$')
plt.ylabel(r'$I$')
plt.show()

plt.plot(deltas/np.pi, Ebonds)
plt.plot([2./3., 2./3.], [0.75, 0.75], 'r.')
plt.xlabel(r'$\delta/\pi$')
plt.ylabel(r'$E$')
plt.show()