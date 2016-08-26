from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Timepoints vector setup:
tf = 2.0    # this is the final time in seconds
steps = 40000
dt = tf/steps
t = np.linspace(0, tf, steps + 1, endpoint=False)

# Oscillator Constants:
# Sinusoidal Constants for delta, theta, alpha, beta, gamma waves:
f = [3.5, 6, 10, 17, 45]    # in Hz
w = [2*np.pi*x for x in f]
# Square Wave Constants:
f_sqr = 2   # in Hz
w_sqr = 2 * np.pi * f_sqr

# Signals:
# Sinusoidal Wave:
wav = [np.cos(x*t) for x in w]
# Square Wave:
sqr = (signal.square(w_sqr * t) + 1) / 2  # runs from 0 to 1, instead of -1 to +1
# Noise:
noise = np.random.normal(0, 0.1, len(t))

# Plot Constants:
sub_rows = 3   # number of subplot rows
sub_cols = 1   # number of subplot columns


# Create Figure
plt.figure()

# First subplot
idx = 1
sgnl = wav
leg = ['Frequency: '+str(x) for x in f]
plt.subplot(sub_rows, sub_cols, idx)
for wave in sgnl:
	plt.plot(t, wave)
plt.ylim(-1.5, 1.5)
plt.legend(leg, loc='upper right')

# Second subplot
idx = 2
sgnl = sqr
plt.subplot(sub_rows, sub_cols, idx)
plt.plot(t, sgnl)
plt.ylim(-0.5, 1.5)

# Third subplot
idx = 3
sgnl = noise
plt.subplot(sub_rows, sub_cols, idx)
plt.plot(t, sgnl)

plt.show()