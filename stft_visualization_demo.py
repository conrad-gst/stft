import numpy as np
import stft_visualization

# generate test signal (linear chirp with noise)
f_start = 1
f_stop = 3
T = 20  # sweep time
T_s = 0.1  # sampling time
t = np.arange(0, T, T_s)
linear_chirp = np.sin(2 * np.pi * (f_start + (f_stop - f_start) / (2 * T) * t) * t)
noise = (2 * np.random.rand(t.size) - 1) * 0.5
linear_chirp_with_noise = linear_chirp + noise


# plot the signal over time
stft_visualization.plot_signal(signal=linear_chirp_with_noise, T_s=T_s)


# plot the dominant frequency of each segment
stft_visualization.plot_dominant_frequencies(
    signal=linear_chirp_with_noise,
    T_s=T_s,
    segment_length=2,
    shift_length=1,
    zero_padding_factor=50,
    window_type="boxcar",
)


# plot the signal, the spectrum of each segment and the dominant frequencies into a single 3d plot
stft_visualization.plot(
    signal=linear_chirp_with_noise,
    T_s=T_s,
    segment_length=2,
    shift_length=1,
    zero_padding_factor=50,
    window_type="boxcar",
    normalize=True,
)
