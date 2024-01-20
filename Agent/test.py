import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
dt = 0.01 # sampling interval
Fs = 1 / dt # sampling frequency
t = np.arange(0, 10, dt) # time vector
# generate noise:
nse = np.random.randn(len(t))
r = np.exp(-t / 0.05)
cnse = np.convolve(nse, r) * dt
cnse = cnse[:len(t)]
s = 0.1 * np.sin(4 * np.pi * t) + cnse # the signal

plt.figure(figsize=(7, 4))
plt.magnitude_spectrum(s, Fs=Fs, scale="dB", color="C1")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Magnitude Spectrum")
plt.savefig("test.png")
# plt.show()
