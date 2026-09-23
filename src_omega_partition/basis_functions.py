"""EDMD dictionary classes used by src_omega_partition.

Vendored subset of ``src/basis_functions.py`` (MPV-SPV-pruning-code); the four classes
below are copied verbatim (numpy only), so lifted features match the original bit-for-bit.
The other bases in that file (Wendland, Legendre, Fourier, inverse multiquadric,
centered polynomial) and its plotting helper are not used here and were left out.

Used by ``src_omega_partition/dictionary.py``. The rest of the
code touches a dictionary object through exactly two members, ``.evaluate(X)`` and ``.s``.

Interface
---------
Every class below has

    .n : int
        State dimension.
    .s : int
        Number of dictionary elements (columns of ``evaluate``).
    .evaluate(X, C_FB=None) -> ndarray, shape (N, s)
        ``X`` has shape (N, n) or (n,) (treated as one state). If ``C_FB`` (shape
        (s, k)) is given, returns ``evaluate(X) @ C_FB`` instead.

class PolynomialBasis(n, d)
    All monomials in ``n`` variables up to total degree ``d``, ordered by total degree
    (``d = 0`` gives only the constant, which is always column 0).

class RadialBasis(n, centers=..., gamma=...)
    Gaussian RBFs ``exp(-gamma * ||x - c||^2)`` for each row ``c`` of ``centers``
    (shape (m, n)); ``gamma`` is a precision, larger = narrower. The original grid mode
    (``d``, ``X_region``), per-center ``gammas`` and ``from_kmeans`` are kept.

class ThinPlateSpline(n, centers)
    Thin-plate splines ``r^2 log r`` with ``r = ||x - c||`` (0 at r = 0; used by
    ``build_thin_plate_dictionary`` only).

class CombinedDictionary(first, second)
    Column-wise concatenation ``[first.evaluate(X), second.evaluate(X)]``; ``s`` is the
    sum of the parts. (The original argument names are ``poly_basis, rbf_basis`` but any
    two bases with matching ``n`` work, in the given order.)
"""

import itertools

import numpy as np
from sklearn.cluster import KMeans


class PolynomialBasis:
    def __init__(self, n, d):
        """
        Parameters:
            n (int): Dimension of input space (number of variables)
            d (int): Maximum total degree of polynomial basis
        """
        self.n = n
        self.d = d
        self.exponents = self._generate_exponents()
        self.s = len(self.exponents)  # Number of basis functions

    def _generate_exponents(self):
        """
        Generates all exponent tuples (e_1, ..., e_n) such that sum(e_i) <= d
        """
        exponents = []
        for total_degree in range(self.d + 1):
            for exp in itertools.product(range(total_degree + 1), repeat=self.n):
                if sum(exp) == total_degree:
                    exponents.append(np.array(exp))
        return exponents

    def evaluate(self, X, C_FB=None):
        """
        Evaluate the full polynomial basis on input data X.
        C_FB is used when we know how to prune the basis functions.

        Parameters:
            X: np.ndarray of shape (n,) or (m, n) — input vector(s)

        Returns:
            Φ: np.ndarray of shape (k,) or (m, k), where k = number of basis functions
               Each row corresponds to basis function evaluations at that input.
        """
        X = np.atleast_2d(X)  # Ensures shape (m, n)
        assert X.shape[1] == self.n, f"Expected input dimension {self.n}, got {X.shape[1]}"

        Phi = []
        for exp in self.exponents:
            Phi.append(np.prod(X ** exp, axis=1))  # shape (m,)

        Phi_X = np.stack(Phi, axis=1)
        if C_FB is None:
            return Phi_X  # shape (m, s)
        else:
            assert len(C_FB.shape) == 2 and C_FB.shape[0] == Phi_X.shape[1]
            return Phi_X @ C_FB

    def get_exponents(self):
        """
        Returns a list of exponent vectors used in the basis
        """
        return self.exponents


class ThinPlateSpline:
    def __init__(self, n, centers):
        """
        Parameters:
            n (int): Dimension of input space (number of variables)
            centers (np.ndarray): Centers for the RBFs, shape (s, n)
        """
        self.n = n
        self.centers = np.atleast_2d(centers)
        assert self.centers.shape[1] == self.n, f"Centers dim {self.centers.shape[1]} != n ({n})"
        self.s = self.centers.shape[0]  # Number of basis functions

    def evaluate(self, X, C_FB=None):
        """
        Evaluate the Thin Plate Spline basis on input data X.
        TPS Kernel: phi(r) = r^2 * ln(r)

        Parameters:
            X: np.ndarray of shape (n,) or (m, n) — input vector(s)
            C_FB: np.ndarray of shape (s, k) — optional projection matrix

        Returns:
            Phi: np.ndarray of shape (m, s) or (m, k)
        """
        X = np.atleast_2d(X)  # Ensures shape (m, n)
        assert X.shape[1] == self.n, f"Expected input dimension {self.n}, got {X.shape[1]}"

        # Compute pairwise Euclidean distances r
        # X: (m, n), Centers: (s, n) -> diff: (m, s, n)
        diff = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        r = np.linalg.norm(diff, axis=2)  # shape (m, s)

        # Calculate phi(r) = r^2 * log(r)
        # We must handle the singularity at r=0 where limit is 0
        Phi_X = np.zeros_like(r)
        mask = r > 1e-10  # Avoid log(0)

        # Apply TPS formula
        Phi_X[mask] = (r[mask] ** 2) * np.log(r[mask])

        if C_FB is None:
            return Phi_X  # shape (m, s)
        else:
            assert len(C_FB.shape) == 2 and C_FB.shape[0] == Phi_X.shape[1]
            return Phi_X @ C_FB

    def get_centers(self):
        """
        Returns the centers used in the basis
        """
        return self.centers


class RadialBasis:
    def __init__(self, n, d=None, gamma=1.0, X_region=None,
                 centers=None, gammas=None):
        """
        Radial Basis Function class.

        Parameters
        ----------
        n : int
            Input dimension.
        d : int, optional
            Number of center points per dimension (for grid mode).
        gamma : float, optional
            Default RBF width if a single gamma is used.
        X_region : ndarray of shape (n, 2), optional
            [x_min, x_max] per dimension, used in grid mode.
        centers : ndarray of shape (s, n), optional
            RBF centers (e.g., from k-means).
        gammas : ndarray of shape (s,), optional
            Per-center gamma values.
        """
        self.n = n

        # If centers are given, use them directly (k-means mode or custom)
        if centers is not None:
            centers = np.asarray(centers)
            assert centers.ndim == 2 and centers.shape[1] == n, \
                f"centers must have shape (s, {n})"
            self.centers = centers
            self.s = centers.shape[0]

            if gammas is None:
                # If no per-center gammas given, use a common one
                self.gammas = np.full(self.s, float(gamma))
            else:
                gammas = np.asarray(gammas, dtype=float)
                assert gammas.shape == (self.s,), \
                    f"gammas must have shape ({self.s},)"
                self.gammas = gammas

            # X_region and d are irrelevant in this mode
            self.X_region = X_region
            self.d = None

        else:
            # Grid-based mode (original behavior)
            assert d is not None, "d must be provided when centers is None"
            self.d = d
            self.gamma = float(gamma)

            if X_region is None:
                self.X_region = np.tile(np.array([[0.0, 1.0]]), (n, 1))
            else:
                X_region = np.asarray(X_region, dtype=float)
                assert X_region.shape == (n, 2), f"X_region must be of shape ({n}, 2)"
                self.X_region = X_region

            self.centers = self._generate_centers()   # (s, n)
            self.s = self.centers.shape[0]
            self.gammas = np.full(self.s, self.gamma)

    def _generate_centers(self):
        """
        Generate RBF centers as a uniform grid over X_region.
        Returns an array of shape (s, n).
        """
        grid_per_dim = [
            np.linspace(self.X_region[i, 0], self.X_region[i, 1], self.d)
            for i in range(self.n)
        ]
        mesh = list(itertools.product(*grid_per_dim))
        return np.array(mesh)  # shape (s, n)

    def evaluate(self, X, C_FB=None):
        """
        Evaluate Gaussian RBFs centered at self.centers.

        Parameters
        ----------
        X : ndarray
            shape (n,) or (m, n)
        C_FB : ndarray, optional
            shape (s, s') projection matrix

        Returns
        -------
        Phi_X : ndarray
            shape (m, s) or (m, s') if C_FB is given
        """
        X = np.atleast_2d(X)
        assert X.shape[1] == self.n, f"Expected input dim {self.n}, got {X.shape[1]}"

        # X: (m, n), centers: (s, n)
        diff = X[:, None, :] - self.centers[None, :, :]  # (m, s, n)
        distsq = np.sum(diff ** 2, axis=2)               # (m, s)

        # per-center gammas: (s,) -> broadcast to (m, s)
        Phi_X = np.exp(-self.gammas[None, :] * distsq)   # (m, s)

        if C_FB is None:
            return Phi_X
        else:
            C_FB = np.asarray(C_FB)
            assert C_FB.shape[0] == Phi_X.shape[1], \
                f"C_FB.shape[0] must be {Phi_X.shape[1]}"
            return Phi_X @ C_FB

    def get_exponents(self):
        """
        Return the array of RBF centers, shape (s, n).
        """
        return self.centers

    def get_gammas(self):
        """
        Return the per-center gamma values, shape (s,).
        """
        return self.gammas

    @classmethod
    def from_kmeans(cls, X, n_centers, random_state=0,
                    min_cluster_size=2, eps=1e-8):
        """
        Construct a RadialBasis instance using k-means clustering on X.

        Parameters
        ----------
        X : ndarray, shape (N, n)
            Data.
        n_centers : int
            Number of RBF centers (k-means clusters).
        random_state : int
            Seed for k-means.
        min_cluster_size : int
            Minimum points in a cluster to trust its local variance.
        eps : float
            Small value to avoid division by zero.

        Returns
        -------
        rbf : RadialBasis
            RBF model with k-means centers and per-center gammas.
        kmeans : KMeans
            Fitted KMeans object.
        """
        X = np.asarray(X)
        N, n = X.shape

        kmeans = KMeans(n_clusters=n_centers, n_init=10, random_state=random_state)
        kmeans.fit(X)

        centers = kmeans.cluster_centers_      # (s, n)
        labels = kmeans.labels_               # (N,)

        s = n_centers
        cluster_var = np.empty(s)
        cluster_var[:] = np.nan

        # cluster-wise mean squared distance to center
        for j in range(s):
            idx = (labels == j)
            pts = X[idx]
            if pts.shape[0] >= min_cluster_size:
                d2 = np.sum((pts - centers[j]) ** 2, axis=1)
                cluster_var[j] = np.mean(d2)
            else:
                # mark as invalid, will fill later
                cluster_var[j] = np.nan

        # Global variance for fallback
        d2_global = np.sum((X - centers[labels]) ** 2, axis=1)
        global_var = np.mean(d2_global)

        # Turn variances into gammas: gamma = 1 / (2 * sigma^2)
        gammas = np.zeros(s)
        for j in range(s):
            var_j = cluster_var[j]
            if not np.isfinite(var_j) or var_j <= 0:
                var_j = global_var
            gammas[j] = 1.0 / (2.0 * (var_j + eps))

        # Build RadialBasis with these centers and gammas
        rbf = cls(n=n, centers=centers, gammas=gammas)
        return rbf, kmeans


class CombinedDictionary:
    def __init__(self, poly_basis, rbf_basis):
        """
        Combines a PolynomialBasis and RadialBasis instance.

        Parameters:
            poly_basis: Instance of PolynomialBasis
            rbf_basis: Instance of RadialBasis
        """
        assert poly_basis.n == rbf_basis.n, "Input dimensions must match"
        self.poly_basis = poly_basis
        self.rbf_basis = rbf_basis
        self.n = poly_basis.n
        self.s = poly_basis.s + rbf_basis.s  # total number of basis functions

    def evaluate(self, X, C_FB=None):
        """
        Evaluate combined dictionary on input X.

        Parameters:
            X (ndarray): shape (n,) or (m, n)
            C_FB (optional): shape (s, s') projection/pruning matrix

        Returns:
            Φ: ndarray of shape (m, s) or (m, s') if C_FB is given
        """
        X = np.atleast_2d(X)
        assert X.shape[1] == self.n, f"Expected input dim {self.n}, got {X.shape[1]}"

        Phi_poly = self.poly_basis.evaluate(X)    # shape (m, s1)
        Phi_rbf = self.rbf_basis.evaluate(X)      # shape (m, s2)
        Phi_combined = np.hstack([Phi_poly, Phi_rbf])  # shape (m, s1 + s2)

        if C_FB is None:
            return Phi_combined
        else:
            assert C_FB.shape[0] == Phi_combined.shape[1]
            return Phi_combined @ C_FB

    def get_exponents(self):
        """
        Returns:
            Tuple of (poly_exponents, rbf_centers)
        """
        return self.poly_basis.get_exponents(), self.rbf_basis.get_exponents()
