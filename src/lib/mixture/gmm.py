import numba as nb
import numpy as np
from sklearn.mixture import GaussianMixture
# from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters


@nb.jit(nb.types.Tuple(
    (nb.float64[:], nb.float64[:, :])
)(nb.float64[:, :], nb.float64[:, :]), nopython=True, nogil=True)
def _estimate_gaussian_parameters(X, resp):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(np.ascontiguousarray(resp.T), X) / np.ascontiguousarray(np.expand_dims(nk, 1))
    return nk, means


class GaussianMixtureIdentity(GaussianMixture):
    def _initialize(self, X, resp):
        n_samples, _ = X.shape
        self.covariances_ = np.zeros(self.n_components)+1.0
        self.precisions_cholesky_ = np.zeros(self.n_components)+1.0
        weights, means = _estimate_gaussian_parameters(X, resp)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        self.covariances_ = np.zeros(self.n_components)+1.0
        self.precisions_cholesky_ = np.zeros(self.n_components)+1.0
        self.weights_, self.means_ = _estimate_gaussian_parameters(X, np.exp(log_resp))
        self.weights_ /= n_samples
