#code from https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html
import numpy as np
from scipy import signal
from scipy.ndimage import filters
from ..util import plot, group_adjacent

def get_novelty_boundaries(S, kernel_size=60, min_dist=25, sigma=4.0):
    novelty = compute_novelty_ssm(S, L=kernel_size, exclude=True)
    return signal.find_peaks(novelty, prominence=0.05)[0]#distance=min_dist)[0]
    #return peak_picking_MSAF(novelty, sigma=sigma)[0]

def binary_boundaries(matrix):
    lastnonzeros = [len(matrix)-np.argmax(m[::-1])-1 for m in matrix]
    maxes = [np.max(lastnonzeros[:i+1]) for i in range(len(lastnonzeros))]
    zeroafters = np.arange(len(matrix))-maxes == 0
    zeroafterbeforetoos = np.arange(len(matrix))-lastnonzeros == 0
    boundaries = prune_boundaries(np.nonzero(zeroafters & zeroafterbeforetoos)[0])+1
    return boundaries[boundaries < len(matrix)]

def discontinuity_boundaries(matrix):
    matrix = np.triu(matrix, k=1)
    indices = [np.nonzero(m)[0] for m in matrix]
    disconts = [len(np.intersect1d(indices[i-1]+1, indices[i])) == 0
        for i in range(1, len(indices))]
    boundaries = np.nonzero(disconts)[0]+1
    return prune_boundaries(boundaries)+1

#takes last of every adjacent group, and first of the last group
def prune_boundaries(boundaries):
    bgroups = group_adjacent(boundaries)
    return np.array([bg[-1] if i < len(bgroups)-1 else bg[0]
        for i,bg in enumerate(bgroups)])

def peak_picking_MSAF(x, median_len=16, offset_rel=0.05, sigma=4.0):
    """Peak picking strategy following MSFA using an adaptive threshold (https://github.com/urinieto/msaf)

    Notebook: C6/C6S1_PeakPicking.ipynb

    Args:
        x (np.ndarray): Input function
        median_len (int): Length of media filter used for adaptive thresholding (Default value = 16)
        offset_rel (float): Additional offset used for adaptive thresholding (Default value = 0.05)
        sigma (float): Variance for Gaussian kernel used for smoothing the novelty function (Default value = 4.0)

    Returns:
        peaks (np.ndarray): Peak positions
        x (np.ndarray): Local threshold
        threshold_local (np.ndarray): Filtered novelty curve
    """
    offset = x.mean() * offset_rel
    x = filters.gaussian_filter1d(x, sigma=sigma)
    threshold_local = filters.median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks, x, threshold_local

def compute_novelty_ssm(S, kernel=None, L=10, var=0.5, exclude=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S (np.ndarray): SSM
        kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
        L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 0.5)
        exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

    Returns:
        nov (np.ndarray): Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov

def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

# print(binary_boundaries(np.array([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])))
# print(discontinuity_boundaries(np.array([[1,0,1,1],[1,1,0,0],[0,0,1,1],[0,0,1,0]])))