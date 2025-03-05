import numpy as np
import cv2

# import matplotlib
# tkagg = matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.ion()


def _validate_arrays(X, Xref, check_2d=False):
    if X.shape != Xref.shape:
        raise ValueError(f"Images should have the same shape, now {X.shape} and {Xref.shape}")
    if not np.any(X):
        raise ValueError("Regridded image is empty")
    if not np.any(Xref):
        raise ValueError("Reference image is empty")
    if check_2d:
        if len(X.shape) != 2:
            raise ValueError("Regridded image should be passed as 2d array for this metric")
        if len(Xref.shape) != 2:
            raise ValueError("Reference image should be passed as 2d array for this metric")


def normalised_difference(X, Xref):
    _validate_arrays(X, Xref)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)
    return np.linalg.norm(X[valid_mask]-Xref[valid_mask]) / np.linalg.norm(Xref[valid_mask])


def root_mean_square_error(X, Xref):
    _validate_arrays(X, Xref)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)

    # Mean of squared differences
    mse = np.mean((X[valid_mask]-Xref[valid_mask])**2)

    return np.sqrt(mse)


def standard_deviation_error(X, Xref):
    _validate_arrays(X, Xref)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)

    diff = X[valid_mask]-Xref[valid_mask]

    return np.std(diff)


def pointwise_correlation(X, Xref):
    _validate_arrays(X, Xref)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)
    return np.corrcoef(X[valid_mask].flatten(), Xref[valid_mask].flatten())[0, 1]


def relative_global_error(X, Xref):
    _validate_arrays(X, Xref)
    return np.linalg.norm((X-Xref)/X)


def total_relative_absolute_error(X, Xref):
    _validate_arrays(X, Xref)
    return np.sum(np.abs((X-Xref)/X))


def mean_absolute_error(X, Xref):
    _validate_arrays(X, Xref)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)
    return np.mean(np.abs(X[valid_mask]-Xref[valid_mask]))

def mean_absolute_percentage_error(X, Xref):
    _validate_arrays(X, Xref)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)
    return 100*np.mean(np.abs((X[valid_mask]-Xref[valid_mask])/Xref[valid_mask]))


def improvement_factor(X, Xref, threshold_dB=-3):
    print("Note: The improvement factor should be calculated only on spot images")
    _validate_arrays(X, Xref)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)
    X = X[valid_mask]
    Xref = Xref[valid_mask]
    Xmax = X.max()
    Xrefmax = Xref.max()
    if not (10*np.log10(Xref/Xrefmax)>=threshold_dB).any():
        return np.nan
    else:
        return np.sqrt(X[10*np.log10(X/Xmax)>=threshold_dB].size / Xref[10*np.log10(Xref/Xrefmax)>=threshold_dB].size)


def peak_error(X, Xref):
    print("Note: The peak error should be calculated only on spot images")
    _validate_arrays(X, Xref)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)
    return Xref[valid_mask].max() - X[valid_mask].max()


def sharpening_factor(X, Xref):

    _validate_arrays(X, Xref, check_2d=True)

    def gradient_magnitude(image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.sum(magnitude)

    Xgrad    = gradient_magnitude(X)
    Xrefgrad = gradient_magnitude(Xref)

    if not np.any(Xrefgrad):
        raise ValueError("Reference image has no gradient, i.e. is a constant image")

    return Xgrad / Xrefgrad


def valid_pixel_overlap(X, Xref):

    _validate_arrays(X, Xref, check_2d=True)
    valid_mask = ~np.isnan(Xref) & ~np.isnan(X)
    min_nan_mask = np.minimum(np.sum(~np.isnan(X)), np.sum(~np.isnan(Xref)))
    # return np.sum(valid_mask) / (0.5 * (np.sum(~np.isnan(X)) + np.sum(~np.isnan(Xref))))
    return np.sum(valid_mask) /  min_nan_mask



    