import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import Mapping, Any, Optional


# https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/mixture/_gaussian_mixture.py
# https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/mixture/_base.py


def _batch_A_T_dot_B(A, B):
    assert A.dim() == B.dim() == 3
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]
    return (A[:, :, :, np.newaxis] * B[:, :, np.newaxis, :]).sum(dim=1)


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.
    Parameters
    ----------
    responsibilities : Tensor, shape (batch_size, n_samples, n_components)
    X : Tensor, shape (batch_size, n_samples, n_features)
    nk : Tensor, shape (batch_size, n_components)
    means : Tensor, shape (batch_size, n_components, n_features)
    reg_covar : float
    Returns
    -------
    covariances : Tensor, shape (batch_size, n_components, n_features)
        The covariance vector of the current components.
    """
    # avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    # avg_means2 = means ** 2
    # avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    # return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

    avg_X2 = _batch_A_T_dot_B(resp, X * X) / nk[..., np.newaxis]
    avg_mean2 = means ** 2
    avg_X_means = means * _batch_A_T_dot_B(resp, X) / nk[..., np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_mean2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(dim=-1)


def _estimate_gaussian_parameters_spherical(X, resp, reg_covar):
    """Estimate the Gaussian distribution parameters.
    Parameters
    ----------
    X : Tensor, shape (batch_size, n_samples, n_features)
        The input data array.
    resp : Tensor, shape (batch_size, n_samples, n_components)
        The responsibilities for each data sample in X.
    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.
    Returns
    -------
    nk : Tensor, shape (batch_size, n_components)
        The numbers of data samples in the current components.
    means : Tensor, shape (batch_size, n_components, n_features)
        The centers of the current components.
    covariances : Tensor (batch_size, n_components)
        The covariance matrix of the current components.
    """
    # nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    # means = np.dot(resp.T, X) / nk[:, np.newaxis]
    # covariances = {"full": _estimate_gaussian_covariances_full,
    #                "tied": _estimate_gaussian_covariances_tied,
    #                "diag": _estimate_gaussian_covariances_diag,
    #                "spherical": _estimate_gaussian_covariances_spherical
    #                }[covariance_type](resp, X, nk, means, reg_covar)
    nk = resp.sum(dim=1) + 10 * torch.finfo(resp.dtype).eps
    means = _batch_A_T_dot_B(resp, X) / nk[..., np.newaxis]
    covariances = _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar)
    return nk, means, covariances


def _estimate_log_gaussian_prob_spherical(X, means, precisions_chol):
    """Estimate the log Gaussian probability.
    Parameters
    ----------
    X : Tensor, shape (batch_size, n_samples, n_features)
    means : Tensor, shape (batch_size, n_components, n_features)
    precisions_chol : Tensor, shape of (batch_size, n_components)
    Returns
    -------
    log_prob : Tensor, shape (batch_size, n_samples, n_components)
    """
    batch_size, n_samples, n_features = X.shape
    # det(precision_chol) is half of det(precision)
    log_det = n_features * torch.log(precisions_chol)

    precisions = precisions_chol ** 2

    # (batch_size, n_samples, n_components)
    log_prob = (
        ((means ** 2).sum(dim=-1) * precisions)[:, np.newaxis, :]
        - (2 * (X[:, :, np.newaxis, :] * (means[:, np.newaxis, :, :] * precisions[:, np.newaxis, :, np.newaxis])).sum(dim=-1))
        + (X ** 2).sum(dim=-1)[:, :, np.newaxis] * precisions[:, np.newaxis, :]
    )
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det[:, np.newaxis, :]


class BatchSphericalGMM:

    def __init__(
        self,
        n_components: int = 1,
        *,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        weights_init: Optional[np.ndarray] = None,
        means_init: Optional[np.ndarray] = None,
        precisions_init: Optional[np.ndarray] = None,
        seed: int = None,
        # warm_start=False,
        # verbose=0,
        # verbose_interval=10,
        centroids_init=None,
        fit_means: bool = True,
        fit_precisions: bool = True,
        kmeans_kwargs: Optional[Mapping[str, Any]] = None,
        device: torch.types.Device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.centroids_init = centroids_init
        self.fit_means = fit_means
        self.fit_precisions = fit_precisions
        self.kmeans_kwargs = {} if kmeans_kwargs is None else kmeans_kwargs
        self.generator = torch.Generator()
        self.device = device
        self.dtype = dtype
        self.seed = seed
        if seed is not None:
            self.generator.manual_seed(seed)

    def fit(self, X: np.ndarray):
        """
        Args:
            X: array, shape (batch_size, n_samples, n_features)
        Returns:
            best_weights: Tensor, (batch_size, n_components)
            best_means: Tensor, (batch_size, n_components)
            best_weights: Tensor, (batch_size, n_components)
        """
        X = torch.from_numpy(X).to(dtype=self.dtype, device=self.device)
        batch_size, n_samples, n_features = X.shape
        max_lower_bound = torch.full((batch_size,), -np.inf, dtype=self.dtype, device=self.device)

        best_weights = torch.full((batch_size, self.n_components), np.nan, dtype=self.dtype, device=self.device)
        best_means = torch.full(
            (batch_size, self.n_components, n_features), np.nan, dtype=self.dtype, device=self.device
        )
        best_covariances = torch.full((batch_size, self.n_components), np.nan, dtype=self.dtype, device=self.device)

        for init in range(self.n_init):
            # self._print_verbose_msg_init_beg(init)

            self._initialize_parameters(X)

            lower_bound = torch.full((batch_size,), -np.inf, dtype=self.dtype, device=self.device)

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                # self._print_verbose_msg_iter_end(n_iter, change)

                if (abs(change) < self.tol).all():
                    self.converged_ = True
                    break

            # self._print_verbose_msg_init_end(lower_bound)

            to_be_updated = lower_bound > max_lower_bound

            best_weights[to_be_updated] = self.weights_[to_be_updated]
            best_means[to_be_updated] = self.means_[to_be_updated]
            best_covariances[to_be_updated] = self.covariances_[to_be_updated]
            max_lower_bound[to_be_updated] = lower_bound[to_be_updated]

        return (
            best_weights.detach().cpu().numpy(),
            best_means.detach().cpu().numpy(),
            best_covariances.detach().cpu().numpy(),
            max_lower_bound.detach().cpu().numpy(),
        )

    def _initialize_parameters(self, X):
        batch_size, n_samples, n_features = X.shape

        if self.centroids_init is not None:
            centroids = torch.from_numpy(self.centroids_init).to(dtype=self.dtype, device=self.device)
            assert centroids.shape == (batch_size, self.n_components, n_features)
            resp = torch.nn.functional.one_hot(
                torch.argmin(((X[:, :, np.newaxis, :] - centroids[:, np.newaxis, :, :]) ** 2).sum(dim=-1), dim=-1),
                num_classes=self.n_components
            ).float()
        elif self.init_params == 'kmeans':
            resp = torch.zeros((batch_size, n_samples, self.n_components), dtype=self.dtype, device=self.device)
            # TODO: batch computation
            for batch_index in range(batch_size):
                label = KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=self.seed, **self.kmeans_kwargs
                ).fit(X[batch_index].detach().cpu().numpy()).labels_
                resp[batch_index, np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = torch.rand((batch_size, n_samples, self.n_components), generator=self.generator)
            resp = resp.to(dtype=self.dtype, device=self.device)
            resp /= resp.sum(dim=1, keepdims=True)
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize(X, resp)

    def _initialize(self, X, resp):
        batch_size, n_samples, n_features = X.shape

        weights, means, covariances = _estimate_gaussian_parameters_spherical(X, resp, self.reg_covar)
        weights /= n_samples

        if self.weights_init is None:
            self.weights_ = weights
        else:
            self.weights_ = torch.from_numpy(self.weights_init).to(dtype=self.dtype, device=self.device)

        if self.means_init is None:
            self.means_ = means
        else:
            self.means_ = torch.from_numpy(self.means_init).to(dtype=self.dtype, device=self.device)

        if self.precisions_init is None:
            self.precisions_cholesky_ = 1. / torch.sqrt(covariances)
            self.covariances_ = covariances
        else:
            self.precisions_cholesky_ = torch.from_numpy(self.precisions_init).to(dtype=self.dtype, device=self.device)
            self.covariances_ = 1 / (self.precisions_cholesky_ ** 2)

    def _e_step(self, X):
        """
        Returns
        -------
        log_prob_norm : Tensor, shape (batch_size,)
            Mean of the logarithms of the probabilities of each sample in X
        log_responsibility : Tensor, shape (batch_size, n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return log_prob_norm.mean(dim=-1), log_resp

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.
        Parameters
        ----------
        X : Tensor, shape (batch_size, n_samples, n_features)
        Returns
        -------
        log_prob_norm : Tensor, shape (batch_size, n_samples)
            log p(X)
        log_responsibilities : Tensor, shape (batch_size, n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=-1)
        # TODO: use a context equivalent to np.errstate(under='ignore')
        log_resp = weighted_log_prob - log_prob_norm[..., np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X):
        """
        Returns
        -------
        weighted_log_prob : Tensor, shape (batch_size, n_samples, n_components)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()[:, np.newaxis, :]

    def _estimate_log_prob(self, X):
        """
        Returns
        -------
        log_prob : Tensor, shape (batch_size, n_samples, n_components)
        """
        return _estimate_log_gaussian_prob_spherical(X, self.means_, self.precisions_cholesky_)

    def _estimate_log_weights(self):
        """
        Returns
        -------
        log_weights : Tensor, shape (batch_size, n_components)
        """
        return torch.log(self.weights_)

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : Tensor, shape (batch_size, n_samples, n_features)
        log_resp : Tensor, shape (batch_size, n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        batch_size, n_samples, n_features = X.shape

        weights, means, covariances = _estimate_gaussian_parameters_spherical(X, torch.exp(log_resp), self.reg_covar)

        self.weights_ = weights / n_samples

        if self.fit_means:
            self.means_ = means

        if self.fit_precisions:
            self.covariances_ = covariances
            self.precisions_cholesky_ = 1. / torch.sqrt(covariances)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm


if __name__ == "__main__":
    from sklearn.mixture import GaussianMixture
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    np_random = np.random.RandomState(0)
    gmm_reference = GaussianMixture(n_components=2, covariance_type="spherical", tol=0, random_state=np_random)
    _initialize_orig = gmm_reference._initialize

    weights_init, means_init, precisions_init = None, None, None

    def _patched_initialize(X, resp):
        global weights_init, means_init, precisions_init
        _initialize_orig(X, resp)
        weights_init = gmm_reference.weights_
        means_init = gmm_reference.means_
        precisions_init = gmm_reference.precisions_cholesky_

    gmm_reference._initialize = _patched_initialize

    batch_size = 32
    n_samples, n_features = 250, 2
    mu1, mu2 = -1.0, 5.0
    sigma1, sigma2 = 1.0, 2.0
    X_batch = []
    weights_init_batch, means_init_batch, precisions_init_batch = [], [], []
    expected_weights, expected_means, expected_covariances = [], [], []
    for _ in range(batch_size):
        n1 = int(n_samples * 0.7) * n_features
        n2 = n_features * n_samples - n1
        X = np_random.normal(np.r_[np.full(n1, mu1), np.full(n2, mu2)], np.r_[np.full(n1, sigma1), np.full(n2, sigma2)])
        X = X.reshape(n_samples, n_features)
        X_batch.append(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gmm_reference.fit(X)

        weights_init_batch.append(weights_init.copy())
        means_init_batch.append(means_init.copy())
        precisions_init_batch.append(precisions_init.copy())

        expected_weights.append(gmm_reference.weights_.copy())
        expected_means.append(gmm_reference.means_.copy())
        expected_covariances.append(gmm_reference.covariances_.copy())

    weights_init_batch = np.asarray(weights_init_batch, dtype=np.float64)
    means_init_batch = np.asarray(means_init_batch, dtype=np.float64)
    precisions_init_batch = np.asarray(precisions_init_batch, dtype=np.float64)
    X_batch = np.asarray(X_batch, dtype=np.float64)

    gmm_tested = BatchSphericalGMM(
        n_components=2,
        weights_init=weights_init_batch,
        means_init=means_init_batch,
        precisions_init=precisions_init_batch,
        init_params="random",
        dtype=torch.float64,
        device="cpu",
    )
    actual_weights, actual_means, actual_covariances = gmm_tested.fit(X_batch)
    assert np.allclose(expected_weights, actual_weights)
    assert np.allclose(expected_means, actual_means)
    assert np.allclose(expected_covariances, actual_covariances)
