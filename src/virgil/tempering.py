"""Tempering an evidence for correlated spectral channels.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

The effective-channel factor ``f = N / N_eff`` is estimated from the
within-chain covariance of the residuals of neighbouring channels, and the
log likelihood is divided by it (``dof_b = 1 / f_b``). The derivation,
the estimator and its limits are in ``docs/api/tempering.md``; the solved
problem is in the AGENTS.md table. This module imports nothing from virgil.
"""

import dataclasses

import numpy as onp

__all__ = [
    "ChannelFactors",
    "FLAG_RANGE",
    "MIN_CHAINS",
    "adjacent_channels",
    "block_dof",
    "channel_chains",
    "estimate_factors",
    "flat_signature",
    "out_of_range",
]

__provisional__ = ("*",)

# Outside this range the result is flagged: the raw ``f`` well below 1 means
# the residuals are anticorrelated (differenced or over-fitted data), the
# largest block ``f`` well above 2 that fewer than half the channels carry
# independent information.
FLAG_RANGE = (0.5, 2.0)
# A block with fewer chains than this takes the pooled f: its own estimate
# scatters by √(2 / chains), 0.45 at 10.
MIN_CHAINS = 10
# ρ2 above ρ1² by more than this (and by 3 standard errors) is not AR(1).
FLAT_EXCESS = 0.15
# Neighbouring samples share a baseline when their (u, v) agree to this many
# metres.
_UV_TOL = 1e-3


@dataclasses.dataclass(frozen=True)
class ChannelFactors:
    """The effective-channel factors of one run.

    Attributes
    ----------
    f : float
        Pooled factor ``N / N_eff`` (at least 1; 1 without chains), used
        where a block has no estimate of its own.
    blocks : dict
        ``{block name: f_b}`` for every block (``"vis"``, ``"phi"``,
        ``"extra[k]"``): its own estimate with ``MIN_CHAINS`` chains, else
        the pooled ``f``; at least 1.
    rho : dict
        ``{block name: ρ1}``, the lag-1 correlation, plus ``"pooled"`` (NaN
        without chains). Diagnostic only.
    rho2 : dict
        The same for the lag-2 correlation ρ2.
    n_pairs : dict
        ``{block name: neighbouring pairs}``, plus ``"pooled"``.
    n_chains : dict
        ``{block name: chains}``, plus ``"pooled"``.
    n_channels : dict
        ``{block name: mean chain length}``, plus ``"pooled"``.
    f_raw : float
        The lowest unfloored ``f̂``; below 1 means anticorrelated residuals
        (flagged, not used).
    raw : dict
        ``{block name: unfloored f̂}`` of the blocks with chains of their
        own, plus ``"pooled"``.
    fallback : tuple of str
        Blocks that took the pooled ``f`` for want of their own estimate.
    n_pairs2 : dict
        ``{block name: pairs two channels apart}``, plus ``"pooled"``: the
        number behind ρ2.
    flat : tuple of str
        The blocks whose lag-2 correlation is not AR(1)'s (see
        ``flat_signature``).
    """

    f: float
    blocks: dict
    rho: dict
    n_pairs: dict
    f_raw: float = 1.0
    n_channels: dict = dataclasses.field(default_factory=dict)
    rho2: dict = dataclasses.field(default_factory=dict)
    n_chains: dict = dataclasses.field(default_factory=dict)
    raw: dict = dataclasses.field(default_factory=dict)
    fallback: tuple = ()
    flat: tuple = ()
    n_pairs2: dict = dataclasses.field(default_factory=dict)

    @property
    def f_max(self):
        """The largest ``f_b``: what ``AliasResult.tempering`` checks."""
        return max([self.f, *self.blocks.values()])

    def block_dof(self):
        """``{block name: 1/f_b}``: the ``block_dof`` of the tempered score."""
        return block_dof(self.blocks)


def block_dof(blocks):
    """The block-keyed ``dof`` that tempers each block by its ``1/f_b``.

    Parameters
    ----------
    blocks : dict
        ``{block name: f_b}``, e.g. ``AliasResult.temper_blocks``
        (``"vis"``, ``"phi"``, ``"extra[k]"``).

    Returns
    -------
    dict
        ``{block name: 1/f_b}`` for the ``block_dof`` argument of
        ``marginal_loglike`` and ``estimate_primary_diameter``, so that a
        later stage (NUTS) reproduces the tempering of the orbit fit. A
        ``f_b`` below 1 is taken as 1: tempering never sharpens.

    Examples
    --------
    >>> block_dof({"vis": 4.0, "phi": 2.0})
    {'vis': 0.25, 'phi': 0.5}
    """
    return {str(k): 1.0 / max(float(v), 1.0) for k, v in dict(blocks).items()}


def channel_chains(data):
    """The chains of consecutive channels of each block of ``data``.

    A chain is the samples that share a baseline (the same ``u``, ``v`` in
    metres) and frame, ordered by wavelength (a closure phase: the same
    triangle in each channel), as indices into the block's residual
    vector. Samples are laid out baseline-major with channels fastest,
    which is how OIData stores several channels, but the chains are found
    from the ``u``, ``v``, ``wavel`` and ``frame`` of each sample, not from
    that order.

    Parameters
    ----------
    data : OIData
        The data (duck-typed: ``u``, ``v``, ``wavel``, ``frame``,
        ``vis``, ``phi``, ``vis_index``, ``phi_index``, ``i_cps1``,
        ``i_cps2``).

    Returns
    -------
    dict
        ``{"vis": [chain, ...], "phi": [chain, ...]}``: each chain an
        integer array of at least two samples with strictly increasing
        wavelength; empty lists for a block with one channel or an
        unreadable layout.
    """
    u = onp.asarray(data.u, float)
    v = onp.asarray(data.v, float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, float), u.shape)
    frame = (
        onp.zeros(u.shape, int)
        if data.frame is None
        else onp.asarray(data.frame)
    )
    n_vis = int(onp.size(data.vis))
    n_phi = int(onp.size(data.phi))
    vis_index = (
        onp.arange(u.size)
        if data.vis_index is None
        else onp.asarray(data.vis_index)
    )
    out = {
        "vis": _chains(_sample_keys(u, v, frame, vis_index), wavel[vis_index])
    }
    if len(vis_index) != n_vis:
        out["vis"] = []
    if not n_phi:
        out["phi"] = []
    elif data.i_cps1 is not None and onp.size(data.i_cps1) == n_phi:
        # A closure phase is keyed by two of its baselines, at the
        # wavelength of the first.
        i1 = onp.asarray(data.i_cps1)
        i2 = onp.asarray(data.i_cps2)
        keys = list(
            zip(_sample_keys(u, v, frame, i1), _sample_keys(u, v, frame, i2))
        )
        out["phi"] = _chains(keys, wavel[i1])
    else:
        index = (
            onp.arange(u.size)
            if data.phi_index is None
            else onp.asarray(data.phi_index)
        )
        out["phi"] = (
            _chains(_sample_keys(u, v, frame, index), wavel[index])
            if len(index) == n_phi
            else []
        )
    return out


def adjacent_channels(data):
    """The neighbouring-channel pairs of each block of ``data``: the
    consecutive samples of ``channel_chains``.

    Returns
    -------
    dict
        ``{"vis": (a, b), "phi": (a, b)}``: integer arrays indexing the
        residual vector of each block, empty when the block has no
        neighbours or its layout cannot be read.
    """
    return {k: _pairs_of(c, 1) for k, c in channel_chains(data).items()}


_EMPTY = (onp.zeros(0, int), onp.zeros(0, int))


def _sample_keys(u, v, frame, index):
    """One hashable key per sample of ``index``: its baseline and frame."""
    q = 1.0 / _UV_TOL
    return [
        (int(round(u[i] * q)), int(round(v[i] * q)), int(frame[i]))
        for i in index
    ]


def _chains(keys, wavel):
    """The wavelength-ordered groups of samples that share a key, with
    repeated wavelengths dropped; groups of one are left out."""
    groups = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    chains = []
    for members in groups.values():
        members = sorted(members, key=lambda p: wavel[p])
        keep = [members[0]]
        for p in members[1:]:
            if wavel[p] > wavel[keep[-1]]:
                keep.append(p)
        if len(keep) > 1:
            chains.append(onp.asarray(keep, int))
    return chains


def _pairs_of(chains, lag):
    """The index pairs ``lag`` channels apart within each chain."""
    a = [c[:-lag] for c in chains if len(c) > lag]
    b = [c[lag:] for c in chains if len(c) > lag]
    if not a:
        return _EMPTY
    return onp.concatenate(a), onp.concatenate(b)


def out_of_range(f, f_raw=None):
    """Whether the factor is outside ``FLAG_RANGE``: ``f`` (the largest
    one used) above its top, or the raw (unfloored) ``f_raw`` below its
    bottom."""
    f_raw = f if f_raw is None else f_raw
    return bool(f > FLAG_RANGE[1] or f_raw < FLAG_RANGE[0])


def flat_signature(rho1, rho2, n_pairs):
    """Whether a lag-2 correlation is not AR(1)'s: ``ρ2 - ρ1²`` above
    ``FLAT_EXCESS`` and three standard errors (1/√n_pairs).

    An AR(1) chain has ρ2 = ρ1². A flat correlation (common mode, a smooth
    model error) has ρ2 close to ρ1, so the excess is ρ1 (1 - ρ1). It is
    the signature of misfit as well as of shared calibration error.
    """
    if not (onp.isfinite(rho1) and onp.isfinite(rho2)) or n_pairs < 1:
        return False
    excess = rho2 - rho1**2
    return bool(excess > max(FLAT_EXCESS, 3.0 / onp.sqrt(n_pairs)))


def _moments(z, chains):
    """``(Σ_chains (Σ z)², Σ z², chains)`` of one block's residuals."""
    z = onp.asarray(z, float)
    num = sum(float(onp.sum(z[c]) ** 2) for c in chains)
    den = sum(float(onp.sum(z[c] ** 2)) for c in chains)
    return num, den, len(chains)


def estimate_factors(residuals, chains, names=()):
    """The effective-channel factors from residuals at the best fit.

    Parameters
    ----------
    residuals : sequence of dict
        One dict per dataset, ``{block name: whitened residuals}``, each
        array in the block's own order (the order ``channel_chains``
        indexes).
    chains : sequence of dict
        One dict per dataset from ``channel_chains``.
    names : sequence of str, optional
        Block names to report beyond those of ``residuals`` (blocks
        without residuals, such as extras, take the pooled ``f``).

    Returns
    -------
    ChannelFactors
        ``f_b = Σ (Σ_chain z)² / Σ z²`` pooled over the chains of block
        ``b`` (an unbiased chain-sum estimator of ``1 + 2 Σ (1 - k/n) ρ_k``,
        recovering the exact ``1 + (n - 1) c`` for common-mode correlation
        ``c``; see `docs/api/tempering.md`); the pooled ``f`` sums all
        blocks. ρ1 and ρ2 are kept as diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.random.default_rng(0).standard_normal(800)
    >>> chains = [np.arange(8 * i, 8 * i + 8) for i in range(100)]
    >>> 1.0 <= estimate_factors([{"vis": z}], [{"vis": chains}]).f < 1.3
    True
    """
    num, den, n_chain, n_len = {}, {}, {}, {}
    lag = {1: {}, 2: {}}  # name -> [Σ z_a z_b, ½ Σ (z_a² + z_b²), pairs]
    for resid, chain in zip(residuals, chains):
        for name, z in resid.items():
            cs = chain.get(name, [])
            if not len(cs):
                continue
            z = onp.asarray(z, float)
            a, b, m = _moments(z, cs)
            num[name] = num.get(name, 0.0) + a
            den[name] = den.get(name, 0.0) + b
            n_chain[name] = n_chain.get(name, 0) + m
            n_len[name] = n_len.get(name, 0) + sum(len(c) for c in cs)
            for k in (1, 2):
                x, y = _pairs_of(cs, k)
                t = lag[k].setdefault(name, [0.0, 0.0, 0])
                t[0] += float(onp.sum(z[x] * z[y]))
                t[1] += 0.5 * float(onp.sum(z[x] ** 2 + z[y] ** 2))
                t[2] += len(x)
    total_num, total_den = sum(num.values()), sum(den.values())
    total_chains = sum(n_chain.values())
    raw_pooled = total_num / total_den if total_den > 0 else float("nan")
    f_pooled = max(raw_pooled, 1.0) if onp.isfinite(raw_pooled) else 1.0

    def corr(table, name):
        t = table.get(name)
        return t[0] / t[1] if t and t[1] > 0 else float("nan")

    pooled = {
        k: (
            sum(t[0] for t in lag[k].values())
            / max(sum(t[1] for t in lag[k].values()), 1e-300)
            if lag[k]
            else float("nan")
        )
        for k in (1, 2)
    }
    everything = set(names) | {n for r in residuals for n in r}
    blocks, raw = {}, {"pooled": raw_pooled}
    rho, rho2 = {"pooled": pooled[1]}, {"pooled": pooled[2]}
    fallback, flat = [], []
    for name in sorted(everything):
        own = n_chain.get(name, 0) >= MIN_CHAINS and den.get(name, 0.0) > 0
        if own:
            raw[name] = num[name] / den[name]
            blocks[name] = max(raw[name], 1.0)
            rho[name], rho2[name] = corr(lag[1], name), corr(lag[2], name)
            if flat_signature(rho[name], rho2[name], lag[2][name][2]):
                flat.append(name)
        else:
            blocks[name] = f_pooled
            fallback.append(name)
    n_lag2 = sum(t[2] for t in lag[2].values())
    if not total_chains:
        fallback = []  # nothing to estimate: not a fallback
    elif (
        not flat
        and len(raw) == 1
        and flat_signature(pooled[1], pooled[2], n_lag2)
    ):
        flat.append("pooled")  # no block of its own: judge all chains
    raws = [v for v in raw.values() if onp.isfinite(v)]
    return ChannelFactors(
        f=f_pooled,
        blocks=blocks,
        rho=rho,
        rho2=rho2,
        n_pairs={
            **{k: v[2] for k, v in lag[1].items()},
            "pooled": sum(t[2] for t in lag[1].values()),
        },
        n_chains={**n_chain, "pooled": total_chains},
        f_raw=min(raws) if raws else 1.0,
        raw=raw,
        n_channels={
            **{k: n_len[k] / n_chain[k] for k in n_chain},
            "pooled": sum(n_len.values()) / max(total_chains, 1),
        },
        fallback=tuple(fallback),
        n_pairs2={**{k: v[2] for k, v in lag[2].items()}, "pooled": n_lag2},
        flat=tuple(flat),
    )
