# STFT Visualization
A python implementation and visualization of a [Short Time Fourier Transform (STFT)](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) using [numpy](https://numpy.org/), [scipy](https://scipy.org/) and [matplotlib](https://matplotlib.org/), with focus on simplicity. 

For computing the STFT, the signal is divided into (eventually overlapping) segments. For each segment, the [DFT](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) is computed (eventually using a window function and using zero padding). The STFT is visualized by plotting the signal, the spectrum of each segment and the frequency of the dominant spectral line of each segment into a single 3d plot. 

The implementation and in particular the visualization is not intended for large data sets, in this case, use the [STFT implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html) coming with scipy and a 2d plot for the spectrogram in which amplitude is encoded in colors (i.e. a heat map).
