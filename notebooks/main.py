import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import (
    welch,
    spectrogram,
    firwin,
    lfilter,
    hilbert,
    butter,
    filtfilt,
    resample_poly,
    windows,
)

t0=26.06 # Change this depending of the actual sample
tmin=t0-2.0
tmax=t0+0.5

# -------------------------------------------------------
# Load audio
# -------------------------------------------------------
filename = "../data/wav/recording_04.wav"

fs, x = wavfile.read(filename)

if x.ndim > 1:
    x = x[:, 0]

x = np.asarray(x, dtype=np.float64)

N_target = int(3e6)

if len(x) < N_target:
    x = np.pad(x, (0, N_target - len(x)))

x += np.random.randn(len(x)) * 1e-3

N = len(x)
t = np.arange(N) / fs


# -------------------------------------------------------
# PSD
# -------------------------------------------------------
fmax=24000

f_psd, Pxx = welch(
    x,
    fs=fs,
    window="hamming",
    nperseg=4096,
    noverlap=2048,
    nfft=8192,
)

plt.figure()
plt.plot(f_psd / 1000, 10 * np.log10(Pxx))
plt.xlabel("Frequency (kHz)")
plt.ylabel("PSD (dB)")
plt.grid(True)


# -------------------------------------------------------
# Spectrogram
# -------------------------------------------------------
f, ts, S = spectrogram(
    x,
    fs=fs,
    window="hamming",
    nperseg=2048,
    noverlap=1024,
    nfft=4096,
    detrend=False,
    scaling="spectrum",
    mode="complex",
)

plt.figure()
plt.pcolormesh(ts, f / 1000, 10 * np.log10(np.abs(S) + 1e-12), shading="auto")
plt.ylim(0, 20)
plt.xlim(0, tmax)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (kHz)")


# -------------------------------------------------------
# Band-pass filter
# -------------------------------------------------------
B = firwin(
    401,
    [1000, 3000],
    pass_zero=False,
    fs=fs,
)

# Filter the signal
y = lfilter(B, [1.0], x)

# Envelope of the filtered signal
a0 = np.abs(hilbert(y))


# -------------------------------------------------------
# Downsample
# -------------------------------------------------------
a1 = resample_poly(a0, up=1, down=100)

fs_env = fs / 100

bhp, ahp = butter(2, 3 / (fs_env / 2), btype="high")

a2 = lfilter(bhp, ahp, a1)

f_env, ts_env, A0 = spectrogram(
    a2,
    fs=fs_env,
    window="hamming",
    nperseg=512,
    noverlap=500,
    nfft=4096,
    detrend=False,
    scaling="spectrum",
    mode="complex",
)

plt.figure()
plt.pcolormesh(
    ts_env,
    f_env,
    10 * np.log10(np.abs(A0) + 0.01),
    shading="auto",
)
plt.ylim(0, 50)
plt.xlim(0, tmax)

# -------------------------------------------------------
# Maximum frequency estimation
# -------------------------------------------------------
PSD = np.abs(A0) ** 2

idx = np.argmax(PSD, axis=0)

Pm = PSD[idx, np.arange(len(idx))]

fmax = f_env[idx]

fmax_smooth = fmax.copy()

df = f_env[1] - f_env[0]

for k in range(len(idx)):
    i = idx[k]

    if 0 < i < len(f_env) - 1:

        ya = PSD[i - 1, k] - Pm[k]
        yb = PSD[i + 1, k] - Pm[k]

        dx = (ya - yb) / (ya + yb) * 0.5

        fmax_smooth[k] += dx * df


plt.figure()
plt.plot(ts_env, fmax, label="FFT")
plt.plot(ts_env, fmax_smooth, label="Interpolated")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(tmin, tmax)
plt.legend()
plt.grid(True)


# -------------------------------------------------------
# Fit power law
# -------------------------------------------------------
t0 = 26.06

cond = ts_env < t0

xfit = np.log10(t0 - ts_env[cond])

lfit = np.log10(fmax_smooth[cond])

slope, intercept = np.polyfit(xfit, lfit, 1)
alpha = -slope
k = 10**intercept

print("alpha =", alpha)
print("k =", k)

f_model = np.zeros_like(ts_env)

f_model[cond] = k * (1 / (t0 - ts_env[cond])) ** alpha

if np.any(~cond):
    f_model[~cond] = f_model[np.where(cond)[0][-1]]

plt.figure()
plt.plot(ts_env, fmax_smooth, label="Measured")
plt.plot(ts_env, f_model, label="Fit")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(tmin, tmax)
plt.title("Power-Law Fit of the Frequency")
plt.legend()
plt.grid(True)


# -------------------------------------------------------
# Phase demodulation
# -------------------------------------------------------
t2 = np.arange(len(a2)) / fs_env

f0 = np.interp(t2, ts_env, f_model)

w0 = 2 * np.pi * f0

phi = np.cumsum(w0) / fs_env

exp_comp = np.exp(-1j * phi)

a2_bb = a2 * exp_comp

blp, alp = butter(2, 1.5 / (fs_env / 2))

a2_f = filtfilt(blp, alp, a2_bb)

phi_i = np.unwrap(np.angle(a2_f))

dw = np.concatenate([np.diff(phi_i) * fs_env, [0]])

f_est = (w0 + dw) / (2 * np.pi)

plt.figure()
plt.plot(ts_env, fmax_smooth, label="Measured")
plt.plot(ts_env, f_model, label="Model")
plt.plot(t2, f_est, label="Estimated")
plt.legend()
plt.grid(True)
plt.xlim(tmin, tmax)

plt.show()