import numpy as np
from typing import Union, Tuple

def rms(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Root mean square along an axis.
    Parameters
    ----------
    x : numpy.ndarray
        The data.
    axis : None or int, default=None
        Axis along which to calculate rms. By default, flatten the array.
    Returns
    -------
    rms : numpy.ndarray
        Root mean square value of x.
    Examples
    --------
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])
    >>> rms(x, axis=0)
    array([ 5.09901951,  9.        ,  5.14781507, 11.04536102])
    >>> rms(x, axis=1)
    array([7.08872344, 8.80340843])
    >>> rms(x)
    7.99218368157289
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(
            f"Argument 'x' must be of type numpy.ndarray, not {type(x).__name__}"
        )
    if not (isinstance(axis, int) | (axis is None)):
        raise TypeError(
            f"Argument 'axis' must be of type int or None, not {type(axis).__name__}"
        )
    return np.sqrt(np.mean(x**2, axis=axis))
def slice_along_axis(arr: np.ndarray, s: slice, axis: int) -> np.ndarray:
    """
    Slice the values of the array within a certain range on the axis.
    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    s : slice
        Range on the `axis`.
    axis : int
        Axis
    Returns
    -------
    arr_out : numpy.ndarray
        Sliced input array.
    """
    arr_out = arr.copy()  # shallow copy
    if axis == -1:
        lower_ndim, upper_ndim = len(arr_out.shape[:axis]), 0
    else:
        lower_ndim, upper_ndim = (
            len(arr_out.shape[:axis]),
            len(arr_out.shape[axis + 1 :]),
        )
    indices = lower_ndim * (np.s_[:],) + (s,) + upper_ndim * (np.s_[:],)
    arr_out = arr_out[indices]
    return arr_out
def positive_fft(
    signal: np.ndarray,
    fs: Union[int, float],
    hann: bool = False,
    normalization: bool = False,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positive 1D fourier transformation.
    Parameters
    ------------
    signal : numpy.ndarray
        Original time-domain signal
    fs : Union[int, float]
        Sampling rate
    hann : bool, default = False
        hann function used to perform Hann smoothing. It is implemented when hann is True
    normalization : bool, default = False
        Normalization after Fourier transform
    axis : int, default=-1
        The axis of the input data array along which to apply the fourier Transformation.
    Returns
    -------
    freq : numpy.ndarray
        frequency
        If input shape is [signal_length,], output shape is freq = [signal_length,].
        If input shape is [n, signal_length,], output shape is freq = [signal_length,].
    mag : numpy.ndarray
        magnitude
        If input shape is [signal_length,], output shape is mag = [signal_length,].
        If input shape is [n, signal_length,], output shape is mag = [n, signal_length,].
    Examples
    --------
    >>> n = 400  # array length
    >>> fs = 800  # Sampling frequency
    >>> t = 1 / fs  # Sample interval time
    >>> x = np.linspace(0.0, n * t, n, endpoint=False) # time
    >>> y = 3 * np.sin(50.0 * 2.0 * np.pi * x) + 2 * np.sin(80.0 * 2.0 * np.pi * x)
    >>> signal = y
    >>> freq, mag = positive_fft(signal, fs,  hann = False, normalization = False, axis = -1)
    >>> freq = np.around(freq[np.where(mag > 1)])
    >>> freq
    [50., 80.]
    """
    if hann is True:
        signal = signal * np.hanning(signal.shape[axis])
    n = signal.shape[axis]
    freq = np.fft.fftfreq(n, d=1 / fs)
    freq = np.abs(freq[: n // 2])
    fft_x = np.fft.fft(signal, axis=axis)
    fft_x_half = slice_along_axis(fft_x, np.s_[: n // 2], axis=axis)
    # Normalization
    if normalization is True:
        mag = np.abs(fft_x_half) / n * 2
    else:
        mag = np.abs(fft_x_half)
    return freq, mag