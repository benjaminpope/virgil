"""Check: the algebra of design/thiele_innes_marginalisation.md.

A NumPy-only check of the note's claims, independent of virgil (it imports
nothing from it). It runs in a few seconds on a laptop::

    uv run --python .venv/bin/python python \
        design/sketches/thiele_innes_marginal_check.py

1. The Jacobian |d(A, B, F, G) / d(a, i, omega, Omega)| = a^3 sin^3 i.
2. (Omega + 180, omega + 180) leaves (A, B, F, G) unchanged.
3. An isotropic Gaussian prior N(0, s^2 I) on (A, B, F, G) implies
   p(cos i) = (1 - cos^2 i) / (1 + cos^2 i)^2, with omega and Omega uniform.
4. The closed-form marginal likelihood (prior-whitened, 4 x 4 Cholesky)
   equals the dense 2n-dimensional Gaussian and a brute-force 4-D
   quadrature, with correlated per-epoch covariances; so do the
   conditional mean and covariance.
5. At e = 0 the marginal likelihood does not depend on the time of
   periastron.
6. Importance weights w = p(a) / (a^3 sin^2 i N(A, B, F, G; 0, s^2 I)),
   per unit of (a, cos i, omega, Omega), turn draws from the Gaussian
   prior into an isotropic orientation with a log-uniform a, and their
   effective sample size is small.
7. At fixed (i, Omega), (A, B, F, G) are linear in the line-of-sight
   constants C = a sin(omega) sin(i) and H = a cos(omega) sin(i), with
   coefficients that diverge as 1 / sin(i) (Wright & Howard 2009, §3.2).
"""

import itertools

import numpy as np

rng = np.random.default_rng(1)


def thiele_innes(a, inc, omega, Omega):
    """(A, B, F, G), angles in radians; ddec = A X + F Y, dra = B X + G Y."""
    cw, sw, cn, sn, ci = (
        np.cos(omega),
        np.sin(omega),
        np.cos(Omega),
        np.sin(Omega),
        np.cos(inc),
    )
    return a * np.array(
        [
            cw * cn - sw * sn * ci,
            cw * sn + sw * cn * ci,
            -sw * cn - cw * sn * ci,
            -sw * sn + cw * cn * ci,
        ]
    )


def campbell_a_cosi(abfg):
    """(a, cos i) from (A, B, F, G), through |z1| and |z2| of the note."""
    A, B, F, G = abfg
    r1 = np.hypot(A + G, B - F)  # a (1 + cos i)
    r2 = np.hypot(A - G, B + F)  # a (1 - cos i)
    return (r1 + r2) / 2, (r1 - r2) / (r1 + r2)


def unit_orbit(t, period, t_peri, ecc):
    """X = cos E - e, Y = sqrt(1 - e^2) sin E, by Newton's method."""
    mean = 2 * np.pi * (t - t_peri) / period
    E = mean + ecc * np.sin(mean)
    for _ in range(50):
        E = E - (E - ecc * np.sin(E) - mean) / (1 - ecc * np.cos(E))
    return np.cos(E) - ecc, np.sqrt(1 - ecc**2) * np.sin(E)


def design_matrix(t, period, t_peri, ecc):
    """Rows (dra, ddec) per epoch, columns (A, B, F, G): shape (n, 2, 4)."""
    x, y = unit_orbit(t, period, t_peri, ecc)
    z = np.zeros_like(x)
    return np.stack(
        [np.stack([z, x, z, y], -1), np.stack([x, z, y, z], -1)], -2
    )


def marginal(t, data, cov, period, t_peri, ecc, mu, prior_sd):
    """log Z, conditional mean and covariance: the note's eq. (M1)-(M3)."""
    chol = np.linalg.cholesky(cov)  # (n, 2, 2)
    whiten = np.linalg.inv(chol)
    D = np.einsum(
        "nij,njk->nik", whiten, design_matrix(t, period, t_peri, ecc)
    )
    D = D.reshape(-1, 4)
    r = np.einsum("nij,nj->ni", whiten, data).reshape(-1) - D @ mu
    S = np.diag(prior_sd)
    K = D @ S
    M = np.eye(4) + K.T @ K
    R = np.linalg.cholesky(M)
    u = np.linalg.solve(R, K.T @ r)
    log_det_c = 2 * np.sum(np.log(np.abs(np.diagonal(chol, 0, 1, 2))))
    log_z = (
        -0.5 * (r @ r - u @ u)
        - np.sum(np.log(np.diag(R)))
        - 0.5 * log_det_c
        - 0.5 * r.size * np.log(2 * np.pi)
    )
    m_inv = np.linalg.inv(M)
    return log_z, mu + S @ m_inv @ K.T @ r, S @ m_inv @ S


def check_jacobian():
    worst = 0.0
    for _ in range(200):
        p = np.array(
            [
                rng.uniform(0.5, 3),
                rng.uniform(0.05, np.pi - 0.05),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
            ]
        )
        h, jac = 1e-6, np.empty((4, 4))
        for k in range(4):
            dp = np.zeros(4)
            dp[k] = h
            jac[:, k] = (thiele_innes(*(p + dp)) - thiele_innes(*(p - dp))) / (
                2 * h
            )
        expected = p[0] ** 3 * np.sin(p[1]) ** 3
        worst = max(worst, abs(abs(np.linalg.det(jac)) / expected - 1))
    print(f"1. Jacobian a^3 sin^3 i: worst relative error {worst:.1e}")
    assert worst < 1e-6


def check_node_symmetry():
    p = rng.uniform(0, 1, 4) * [2, np.pi, 2 * np.pi, 2 * np.pi]
    same = thiele_innes(p[0], p[1], p[2] + np.pi, p[3] + np.pi)
    err = np.max(np.abs(same - thiele_innes(*p)))
    print(f"2. (Omega + pi, omega + pi) invariance: max difference {err:.1e}")
    assert err < 1e-12


def check_implied_prior():
    a, c = campbell_a_cosi(rng.normal(0, 1, (4, 2_000_000)))
    edges = np.linspace(-1, 1, 21)
    hist, _ = np.histogram(c, edges, density=True)
    mid = (edges[1:] + edges[:-1]) / 2
    # Bin averages of p(c) = (1 - c^2)/(1 + c^2)^2 = d/dc [c / (1 + c^2)].
    cdf = edges / (1 + edges**2)
    expected = np.diff(cdf) / np.diff(edges)
    err = np.max(np.abs(hist - expected))
    face_on = np.mean(np.abs(c) > np.cos(np.deg2rad(30)))
    exact = 1 - np.sin(2 * np.arctan(np.cos(np.deg2rad(30))))
    print(
        "3. p(cos i) = (1 - c^2)/(1 + c^2)^2: max histogram error "
        f"{err:.3f} (density ~1 at c = {mid[10]:.2f}); P(i < 30 or > 150) "
        f"= {face_on:.4f} (exact {exact:.4f}; isotropic "
        f"{1 - np.cos(np.deg2rad(30)):.4f})"
    )
    assert err < 0.01 and abs(face_on - exact) < 1e-3


def simulate(n=6, ecc=0.4):
    t = np.sort(rng.uniform(0, 800, n))
    truth = thiele_innes(20.0, np.deg2rad(50), 1.0, 2.0)
    D = design_matrix(t, 1000.0, 130.0, ecc)
    # Correlated per-epoch errors (mas^2).
    L = rng.normal(0, 1, (n, 2, 2)) * 0.6 + np.eye(2)
    cov = L @ L.swapaxes(-1, -2)
    noise = np.einsum(
        "nij,nj->ni", np.linalg.cholesky(cov), rng.normal(size=(n, 2))
    )
    return t, D @ truth + noise, cov


def check_marginal():
    t, data, cov = simulate()
    mu, sd = (
        np.array([1.0, -2.0, 0.5, 3.0]),
        np.array([30.0, 25.0, 30.0, 20.0]),
    )
    log_z, mean, post_cov = marginal(t, data, cov, 1000.0, 130.0, 0.4, mu, sd)
    # Dense: d ~ N(D mu, C + D Lambda D^T), with C block diagonal.
    D = design_matrix(t, 1000.0, 130.0, 0.4).reshape(-1, 4)
    n2 = D.shape[0]
    C = np.zeros((n2, n2))
    for k in range(len(t)):
        C[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = cov[k]
    total = C + D @ np.diag(sd**2) @ D.T
    r = data.reshape(-1) - D @ mu
    dense = -0.5 * (
        r @ np.linalg.solve(total, r) + np.linalg.slogdet(2 * np.pi * total)[1]
    )
    # Brute force: a tensor grid over +-8 sd along the posterior's axes.
    vals, vecs = np.linalg.eigh(post_cov)
    nodes = np.linspace(-8, 8, 33)
    step = nodes[1] - nodes[0]
    grid = np.array(list(itertools.product(nodes, repeat=4)))
    theta = mean + (grid * np.sqrt(vals)) @ vecs.T
    resid = data.reshape(-1)[None] - theta @ D.T
    c_inv = np.linalg.inv(C)
    log_like = -0.5 * np.einsum("ki,ij,kj->k", resid, c_inv, resid)
    log_like -= 0.5 * np.linalg.slogdet(2 * np.pi * C)[1]
    dev = (theta - mu) / sd
    log_prior = (
        -0.5 * np.sum(dev**2, 1) - np.sum(np.log(sd)) - 2 * np.log(2 * np.pi)
    )
    log_f = log_like + log_prior
    peak = log_f.max()
    weight = np.exp(log_f - peak)
    volume = step**4 * np.prod(np.sqrt(vals))
    brute = peak + np.log(weight.sum() * volume)
    w = weight / weight.sum()
    grid_mean = w @ theta
    grid_cov = (theta - grid_mean).T @ ((theta - grid_mean) * w[:, None])
    print(
        f"4. log Z: closed form {log_z:.6f}, dense {dense:.6f}, "
        f"quadrature {brute:.6f}; conditional mean error "
        f"{np.max(np.abs(grid_mean - mean)):.1e} mas, covariance relative "
        f"error {np.max(np.abs(grid_cov / post_cov - 1)):.1e}"
    )
    assert abs(log_z - dense) < 1e-8 and abs(log_z - brute) < 1e-5


def check_circular():
    t, data, cov = simulate(ecc=0.0)
    mu, sd = np.zeros(4), np.full(4, 30.0)
    values = [
        marginal(t, data, cov, 1000.0, tp, 0.0, mu, sd)[0]
        for tp in np.linspace(0, 1000, 7)
    ]
    spread = np.ptp(values)
    print(f"5. e = 0: log Z over 7 times of periastron spans {spread:.1e}")
    assert spread < 1e-9


def check_reweighting(s=1.0, a_lo=0.3, a_hi=3.0):
    abfg = rng.normal(0, s, (4, 2_000_000))
    a, c = campbell_a_cosi(abfg)
    gauss = np.exp(-np.sum(abfg**2, 0) / (2 * s**2))
    inside = (a > a_lo) & (a < a_hi)
    w = np.where(inside, 1 / (a * a**3 * (1 - c**2) * gauss), 0.0)
    w /= w.sum()
    hist_c, _ = np.histogram(c, np.linspace(-1, 1, 11), weights=w)
    hist_a, _ = np.histogram(
        np.log(a), np.linspace(np.log(a_lo), np.log(a_hi), 11), weights=w
    )
    ess = 1 / np.sum(w**2) / w.size
    print(
        "6. reweighted to isotropic, log-uniform a: cos i bins "
        f"{hist_c.min():.3f}-{hist_c.max():.3f} and log a bins "
        f"{hist_a.min():.3f}-{hist_a.max():.3f} (0.1 each); effective "
        f"sample size {100 * ess:.2f}% of the draws"
    )
    assert np.all(np.abs(hist_c - 0.1) < 0.02)
    assert np.all(np.abs(hist_a - 0.1) < 0.02)


def check_line_of_sight_form():
    worst = 0.0
    for _ in range(100):
        a, i, w, n = rng.uniform(0, 1, 4) * [3, np.pi, 2 * np.pi, 2 * np.pi]
        C, H = a * np.sin(w) * np.sin(i), a * np.cos(w) * np.sin(i)
        cn, sn, ci, si = np.cos(n), np.sin(n), np.cos(i), np.sin(i)
        linear = (
            np.array(
                [
                    H * cn - C * sn * ci,
                    H * sn + C * cn * ci,
                    -C * cn - H * sn * ci,
                    -C * sn + H * cn * ci,
                ]
            )
            / si
        )
        worst = max(worst, np.max(np.abs(linear - thiele_innes(a, i, w, n))))
    print(f"7. (A, B, F, G) linear in (C, H) at fixed (i, Omega): {worst:.1e}")
    assert worst < 1e-9


if __name__ == "__main__":
    check_jacobian()
    check_node_symmetry()
    check_implied_prior()
    check_marginal()
    check_circular()
    check_reweighting()
    check_line_of_sight_form()
