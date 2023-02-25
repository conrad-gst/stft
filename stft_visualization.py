import numpy as np
import scipy
import matplotlib.pyplot as plt


def slice_signal(signal:np.ndarray, T_s=1.0, segment_length=None, shift_length=1.0):
    """
    Slices the signal into (eventually overlapping) segments of length segment_length.
    The segments overlap if shift_length < segment_length.

    Parameters
    ----------
    signal : numpy.ndarray
        1D array containing the signal samples
    T_s : float, optional
        sampling time of the signal (default is 1.0 seconds)
    segment_length : float, optional
        length of the segments (default is None, meaning that the whole signal itself 
        is the only segment)
    shift_length : float
        length by which the beginnings of the segments are separated (default is 1.0 
        seconds)

    Returns
    -------
    signal_segments : numpy.ndarray
        2D array whose rows contain the segments
    segment_centers : numpy.ndarray
        1D array containing the center time of each segment  
    """

    if (segment_length is None):
        segment_length = signal.size
    else:
        segment_length = round(segment_length/T_s)
        
    shift_length = round(shift_length/T_s)
    num_segments = int((signal.size - segment_length) / shift_length) + 1
    segment_centers = (np.arange(0, num_segments) * shift_length + segment_length / 2) * T_s
    signal_segments = np.zeros((num_segments, segment_length))
    for i in range(num_segments):
        signal_segments[i, 0:segment_length] = signal[i * shift_length : i * shift_length + segment_length]

    return signal_segments, segment_centers


def compute_stft(signal:np.ndarray, T_s=1.0, segment_length=None, shift_length=1.0, zero_padding_factor=0.0, window_type='boxcar'):
    """
    Compute a Short Time Fourier Transform (STFT), i.e. slices the signal into (eventually overlapping) segments of length 
    segment_length (the segments overlap if shift_length < segment_length) and compute the DFT for each segment, eventually 
    using a window function and using zero padding.

    Parameters
    ----------
    signal : numpy.ndarray
        1D array containing the signal samples
    T_s : float, optional
        sampling time of the signal (default is 1.0 seconds)
    segment_length : float, optional
        length of the segments (default is None, meaning that the whole signal itself 
        is the only segment)
    shift_length : float
        length by which the beginnings of the segments are separated (default is 1.0 
        seconds)
    zero_padding_factor : float, optional
        factor by which the length of the signal is extended with zeros (default is 0, which 
        means that no zeros are added) 
    window_type : str
        type of the used window (default is a 'boxcar' (i.e. rectangular) window)

    Returns
    -------
    f : numpy.ndarray
        1D array containing the DFT sample frequencies
    segment_centers : numpy.ndarray
        1D array containing the center time of each segment
    signal_segments_spec : numpy.ndarray
        2D array whose rows contain amplitude spectra of the signal segments
    f_dominant : numpy.ndarray
        1D array containing for each segment the DFT sample frequency of the dominant spectral line
    """

    signal_segments, segment_centers = slice_signal(signal, T_s, segment_length, shift_length)
    num_rows, num_cols = np.shape(signal_segments)
    zeros = np.zeros((num_rows, round(num_cols * zero_padding_factor)))
    window = scipy.signal.windows.get_window(window_type, num_cols)
    signal_segments = signal_segments * window
    signal_segments_padded = np.column_stack((signal_segments, zeros))
    signal_segments_spec = np.abs(scipy.fft.rfft(signal_segments_padded))
    f = scipy.fft.rfftfreq(round(num_cols * (1 + zero_padding_factor)) , d=T_s)
    signal_segments_spec_peak_idx = np.argmax(signal_segments_spec, axis=1)
    f_dominant = f[signal_segments_spec_peak_idx]
    return f, segment_centers, signal_segments_spec, f_dominant


def plot_signal(signal, T_s=1.0):
    t = np.arange(0, signal.size) * T_s
    _, axis = plt.subplots(1)
    axis.plot(t, signal)
    axis.set_xlabel(r"$t$ in seconds")
    axis.set_ylabel(r"signal")
    axis.grid()
    plt.show()


def plot_dominant_frequencies(signal:np.ndarray, T_s=1.0, segment_length=None, shift_length=1.0, zero_padding_factor=0.0, window_type='boxcar'):
    """
    Slices the signal into (eventually overlapping) segments of length segment_length (the segments overlap if shift_length < segment_length)
    and compute the DFT for each segment, eventually using a window function and using zero padding. In the spectrum of each segment, the 
    dominant frequency is searched and plotted over time (namely the center time of each segment).

    Parameters
    ----------
    signal : numpy.ndarray
        1D array containing the signal samples
    T_s : float, optional
        sampling time of the signal (default is 1.0 seconds)
    segment_length : float, optional
        length of the slices (default is None, meaning that the whole signal itself is the only slice)
    shift_length : float
        length by which the beginnings of the slices are separated (default is 1.0 seconds)
    zero_padding_factor : float, optional
        factor by which the signal slices are extended with zeros for extending frequency resolution
        (default is 0, which means that no zeros are added)
    window_type : str
        type of the used window (default is a 'boxcar' (i.e. rectangular) window)
    """

    _, segment_centers, _, f_dominant = compute_stft(signal, T_s, segment_length, shift_length, zero_padding_factor, window_type)
    _, axis = plt.subplots(1)
    axis.plot(segment_centers, f_dominant)
    axis.set_xlabel(r"$t$ in seconds")
    axis.set_ylabel(r"$f$ in Hz")
    axis.grid()
    plt.show()


def plot(signal:np.ndarray, T_s=1.0, segment_length=None, shift_length=1.0, zero_padding_factor=0.0, window_type='boxcar', normalize=True):
    """
    Slices the signal into (eventually overlapping) segments of length segment_length (the segments overlap if shift_length < segment_length)
    and compute the DFT for each segment, eventually using a window function and using zero padding. Plot the signal, the amplitude
    spectrum of each segment and the dominant frequency of each segment into a single 3d plot.

    Parameters
    ----------
    signal : numpy.ndarray
        1D array containing the signal samples
    T_s : float, optional
        sampling time of the signal (default is 1.0 seconds)
    slice_length : float, optional
        length of the slices (default is None, meaning that the whole signal itself 
        is the only slice)
    shift_length : float
        length by which the beginnings of the slices are separated (default is 1.0 seconds)
    zero_padding_factor : float, optional
        factor by which the signal slices are extended with zeros for extending frequency resolution
        (default is 0, which means that no zeros are added)
    window_type : str
        type of the used window (default is a 'boxcar' (i.e. rectangular) window)
    normalize : boolean
        if True, normalize signal to [-1,1] and the amplitude spectrum to [0,1] (default is True)      
    """
    
    f, segment_centers, signal_segments_spec, f_dominant = compute_stft(signal, T_s, segment_length, shift_length, zero_padding_factor, window_type)

    _, axis = plt.subplots(subplot_kw={"projection": "3d"})
    t = np.arange(0, signal.size) * T_s
    X_grid, Y_grid = np.meshgrid(f, segment_centers)
    signal_normalization_factor = 1
    spectrum_normalization_factor = 1
    
    if normalize:
        signal_normalization_factor = np.max(np.abs(signal))
        spectrum_normalization_factor = np.max(signal_segments_spec)
    
    axis.plot(np.zeros(signal.size), t, signal / signal_normalization_factor)
    axis.plot_wireframe(X_grid, Y_grid, signal_segments_spec / spectrum_normalization_factor, cstride=0, color="g")
    axis.plot(f_dominant, segment_centers, np.zeros(segment_centers.size), color="r")

    axis.set_xlabel(r"$f$ in Hz")
    axis.set_ylabel(r"$t$ in seconds")
    if normalize:
        axis.set_zlabel(r"normalized amplitude")
    else:
        axis.set_zlabel(r"amplitude")
    plt.show()