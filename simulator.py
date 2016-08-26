from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Timepoints vector setup:
tf = 40.0    # this is the final time in seconds
steps = 40000
dt = tf/steps
t = np.linspace(0, tf, steps + 1, endpoint=False)

# Oscillator Constants:
# Sinusoidal Constants:
f = 1    # in Hz
w = 2 * np.pi * f
# Square Wave Constants:
f_sqr = 0.25   # in Hz
w_sqr = 2 * np.pi * f_sqr

# Signals:
# Sinusoidal Wave:
wav = np.cos(w * t)
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
plt.subplot(sub_rows, sub_cols, idx)
plt.plot(t, sgnl)
plt.ylim(-1.5, 1.5)

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