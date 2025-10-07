# mc_uncert.py
import numpy as np

def simulate_v_from_V(
    V_mean,
    a_asc,
    sigma_V,
    sigma_lackofit,
    Cov_a=None,
    include_param_cov=False,
    N=100_000,
    rng=None,
):
    """
    Monte Carlo for v = a0 + a1 V + a2 V^2 + a3 V^3 + a4 V^4

    Parameters
    ----------
    V_mean : float
        Measured mean voltage (e.g., from LVM).
    a_asc : array-like, shape (5,)
        Polynomial coefficients [a0, a1, a2, a3, a4] (ascending V powers).
    sigma_V : float
        1-sigma voltage noise (std dev of the voltage series).
    sigma_lackofit : float
        1-sigma model residual in velocity units (your “residual σ” from the fit).
    Cov_a : (5,5) array or None
        Covariance of coefficients (ascending order). Required if include_param_cov=True.
    include_param_cov : bool
        If True, sample a ~ N(a, Cov_a) each trial. If False, keep a fixed.
    N : int
        Number of Monte Carlo trials.
    rng : np.random.Generator or None
        For reproducibility.

    Returns
    -------
    stats : dict
        {"mean": float, "std": float, "p05": float, "p50": float, "p95": float}
    samples : np.ndarray
        The N velocity samples (so you can inspect the distribution if desired).
    """
    rng = np.random.default_rng() if rng is None else rng
    a = np.asarray(a_asc, dtype=float)

    if include_param_cov:
        if Cov_a is None:
            raise ValueError("Cov_a must be provided when include_param_cov=True")
        A = rng.multivariate_normal(mean=a, cov=np.asarray(Cov_a), size=N)  # (N,5)
    else:
        A = np.tile(a, (N, 1))  # (N,5) all rows identical

    # Sample voltages and model lack-of-fit noise (independent, zero-mean)
    V_samp = rng.normal(loc=V_mean, scale=sigma_V, size=N)
    eps = rng.normal(loc=0.0, scale=sigma_lackofit, size=N)

    # Evaluate polynomial for each row of A at V_samp:
    # v = a0 + a1 V + a2 V^2 + a3 V^3 + a4 V^4
    Vp = np.vstack([np.ones_like(V_samp),
                    V_samp,
                    V_samp**2,
                    V_samp**3,
                    V_samp**4]).T  # shape (N,5)
    v_samp = np.einsum("ij,ij->i", A, Vp) + eps

    stats = {
        "mean": float(v_samp.mean()),
        "std": float(v_samp.std(ddof=1)),
        "p05": float(np.percentile(v_samp, 5)),
        "p50": float(np.percentile(v_samp, 50)),
        "p95": float(np.percentile(v_samp, 95)),
    }
    return stats, v_samp
