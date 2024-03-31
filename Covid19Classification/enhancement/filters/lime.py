# Implementation adapted from: https://github.com/pvnieo/Low-light-Image-Enhancement

import numpy as np
import cv2
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

lime_config = {
    "gamma": 0.3,
    "alpha": 0.1,
    "sigma": 3,
    "epsilon": 0.001,
    "optimal_dimensions": (200, 200)
}


def get_sparse_neighbor(p: int, n: int, m: int) -> dict:
    """Returns a dictionnary, where the keys are index of 4-neighbor of `p` in the sparse matrix,
       and values are tuples (i, j, x), where `i`, `j` are index of neighbor in the normal matrix,
       and x is the direction of neighbor.

    Arguments:
        p {int} -- index in the sparse matrix.
        n {int} -- number of rows in the original matrix (non sparse).
        m {int} -- number of columns in the original matrix.

    Returns:
        dict -- dictionnary containing indices of 4-neighbors of `p`.
    """
    i, j = p // m, p % m
    d = {}
    if i - 1 >= 0:
        d[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        d[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        d[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        d[i * m + j + 1] = (i, j + 1, 1)
    return d


def create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15) -> np.ndarray:
    """Create a kernel (`size` * `size` matrix) that will be used to compute the he spatial affinity based Gaussian weights.

    Arguments:
        spatial_sigma {float} -- Spatial standard deviation.

    Keyword Arguments:
        size {int} -- size of the kernel. (default: {15})

    Returns:
        np.ndarray - `size` * `size` kernel
    """
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j),
                                  (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

    return kernel


def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Compute the smoothness weights used in refining the illumination map optimization problem.

    Arguments:
        L {np.ndarray} -- the initial illumination map to be refined.
        x {int} -- the direction of the weights. Can either be x=1 for horizontal or x=0 for vertical.
        kernel {np.ndarray} -- spatial affinity matrix

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability. (default: {1e-3})

    Returns:
        np.ndarray - smoothness weights according to direction x. same dimension as `L`.
    """
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode='constant')
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    return T / (np.abs(Lp) + eps)

def refine_illumination_map_linear(L: np.ndarray, gamma: float, lambda_: float,
                                   kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Refine the illumination map based on the optimization problem described in the two papers.
       This function use the sped-up solver presented in the LIME paper.

    Arguments:
        L {np.ndarray} -- the illumination map to be refined.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3}).

    Returns:
        np.ndarray -- refined illumination map. same shape as `L`.
    """
    # compute smoothness weights
    wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)

    n, m = L.shape
    L_1d = L.copy().flatten()

    # compute the five-point spatially inhomogeneous Laplacian matrix
    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    # solve the linear system
    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

    # gamma correction
    L_refined = np.clip(L_refined, eps, 1) ** gamma

    return L_refined


def correct_underexposure(image: np.ndarray, gamma: float, lambda_: float, 
                          kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """correct underexposudness using the retinex based algorithm presented in DUAL and LIME paper.

    Arguments:
        image {np.ndarray} -- input image to be corrected.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3})

    Returns:
        np.ndarray -- image underexposudness corrected. same shape as `im`.
    """

    # first estimation of the illumination map
    L = np.max(image, axis=-1)
    # illumination refinement
    L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)

    # correct image underexposure
    L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
    image_corrected = image / L_refined_3d
    return image_corrected


def lime(image: np.ndarray) -> np.ndarray:
    """Enhance input image, using LIME method.

    Arguments:
        image {np.ndarray} -- input image to be corrected.

    Returns:
        np.ndarray -- image exposure enhanced. same shape as `image`.
    """
    # create spacial affinity kernel
    kernel = create_spacial_affinity_kernel(lime_config["sigma"])

    original_shape = image.shape
    if lime_config["optimal_dimensions"] is not None:
        dimensions = lime_config["optimal_dimensions"][:2]
        image = cv2.resize(image, dimensions, interpolation=cv2.INTER_CUBIC)

    image_normalized = image.astype(float) / 255.0
    if len(image_normalized.shape) == 2:
        image_normalized = image_normalized[..., np.newaxis]

    # correct underexposudness
    under_corrected = correct_underexposure(
        image_normalized, lime_config["gamma"], lime_config["alpha"], kernel, lime_config["epsilon"])

    image_corrected = under_corrected

    clipped_image = np.clip(image_corrected * 255, 0, 255).astype("uint8")

    if lime_config["optimal_dimensions"] is not None:
        clipped_image = cv2.resize(
            clipped_image, original_shape[:2], interpolation=cv2.INTER_CUBIC
        )
    # convert to 8 bits and returns
    return clipped_image
