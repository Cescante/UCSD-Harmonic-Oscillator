from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Timepoints vector setup:
tf = 16.0    # this is the final time in seconds
steps = 40000
dt = tf/steps
t = np.linspace(0, tf, steps + 1, endpoint=False)

# Oscillator Constants:
k = 0.1
m = 0.01
w = (k / m)**0.5
f = w/(2*np.pi)  # in Hz
b = (w/50)*0.1   # damping set to 10% of critical damping
# Square Wave Constants:
f_sqr = 2   # in Hz
w_sqr = 2 * np.pi * f_sqr

# Signals:
# Sinusoidal Wave:
# Calculated using Euler's Method
# Solving: x'' = -w^2 x, where x = cos(w*t)
pos = np.zeros(steps + 1)
vel = np.zeros(steps + 1)
pos[0] = 0.01
vel[0] = 0.05
for i in range(0, steps):
	vel[i+1] = (-w * dt * pos[i]) + (vel[i] * (1 - (dt * b/m)))
	pos[i+1] = dt*vel[i] + pos[i]
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
sgnl = pos
plt.subplot(sub_rows, sub_cols, idx)
plt.plot(t, sgnl)
#plt.ylim(-1.5, 1.5)

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