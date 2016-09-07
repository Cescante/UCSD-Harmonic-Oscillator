from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Timepoints vector setup:
tf = 100.0    # this is the final time in seconds
steps = 400000
dt = tf/steps
t = np.linspace(0, tf, steps + 1, endpoint=False)

# Oscillator Constants:
k = 0.1
m = 0.01
w = (k / m)**0.5
f = w/(2*np.pi)  # in Hz
b = 0   # damping set to 0
# Square Wave Constants:
f_sqr = .1   # in Hz
w_sqr = 2 * np.pi * f_sqr
# Driving AC Current Sinusoidal Constants:
#f_drv = 7.0
w_drv = 25.0  #2 * np.pi * f_drv
F0 = 0.001    # Amplitude

# Signals:
# Uneven Square Wave:
pulse_width = 2000
cycle_width = 10000
sqr = np.zeros(steps + 1)
for i in range(0, steps + 1):
	if (i % cycle_width) < pulse_width:
		sqr[i] = 1
	else:
		sqr[i] = 0

# Driven Wave:
# Calculated using Euler's Method
# Acceleration due to damping force (proportional to velocity):
#     b * v = m * a  -->  a = (b/m) * v
# Acceleration due to driving force:
#     curr = m * a  -->  a = curr/m
# Total acceleration (F is driving force):
# a(t) = v'(t) = -w^2 * x(t) - (b/m) * v(t) + F/m
# Rewrite derivative: v'(t) = (v(t + dt) - v(t))/dt
# (v(t + dt) - v(t))/dt = -w^2 * x(t) - (b/m) * v(t) + F/m
# v(t + dt) = -w^2 * x(t) * dt + v(t) * (1 - (dt * b/m)) + F * dt/m
# Similarly:
# x'(t) = (x(t + dt) - x(t))/dt = v(t)
# x(t + dt) = v(t) * dt + x(t)
pos = np.zeros(steps + 1)
vel = np.zeros(steps + 1)
pos[0] = 0.01
vel[0] = 0.05
for i in range(0, steps):
	vel[i+1] = (-w * w * dt * pos[i]) + (vel[i] * (1 - (dt * b/m))) + ((dt/m) * F0 * np.cos(w_drv*i*dt))
	pos[i+1] = dt*vel[i] + pos[i]


# Damped Driven Wave:
# b was originally set to 0, we can now set it to 10% of the critical damping
b = (w/50)*0.1
pos_d = np.zeros(steps + 1)
pos_d[0] = 0.01
for i in range(0, steps):
	vel[i+1] = (-w * w * dt * pos_d[i]) + (vel[i] * (1 - (dt * b/m))) + ((dt/m) * F0 * np.cos(w_drv*i*dt))
	pos_d[i+1] = dt*vel[i] + pos_d[i]

# Damped Driven Wave (driven by square AC pulses):
pos_ds = np.zeros(steps + 1)
pos_ds[0] = 0.01
for i in range(0, steps):
	vel[i+1] = (-w * w * dt * pos_ds[i]) + (vel[i] * (1 - (dt * b/m))) + ((dt/m) * F0 * np.cos(w_drv*i*dt) * sqr[i])
	pos_ds[i+1] = dt*vel[i] + pos_ds[i]

# Noise:
noise = np.random.normal(0, 0.1, len(t))

# x-axis
y = np.zeros(steps + 1)

# Create Figure 1
# Figure 1 displays 1 oscillator driven by a constant AC current
plt.figure()
plt.subplot(3, 1, 1)
#plt.plot(t, pos, label='Undamped')
plt.plot(t, pos_d, label='10% of Critical Damping, Driven by Constant AC Current')
plt.plot(t, y)
plt.legend(loc='upper right')

plt.subplot(3, 1, 2)
plt.plot(t, pos_ds, label='10% of Critical Damping, Driven by Square AC Pulses')
plt.plot(t, y)
plt.legend(loc='upper right')

plt.subplot(3, 1, 3)
#plt.plot(t, F0 * np.cos(w_drv*t), label='Driving Force')
plt.plot(t, sqr * F0 * np.cos(w_drv*t), label='Driving Force, Pulses')
plt.ylim(-0.0015, 0.0015)
plt.legend(loc='upper right')
plt.show()