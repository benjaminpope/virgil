"""Named epochs of data, with one snapshot of a moving scene per dataset.

An orbit fit to interferometric data across epochs predicts each epoch's
visibilities and closure phases from the scene at that epoch's time. When
nothing moves appreciably during an observation (a binary whose period is
much longer than a night), one snapshot of the scene per dataset is
enough: the orbit is solved once per dataset, and the snapshot is a static
model, evaluated on all of the dataset's samples at once on the fast path
(e.g. [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian] for
[`OrbitalBinary`][virgil.models.OrbitalBinary]). That is much cheaper
than [`OIData.model`][virgil.oidata.OIData.model]'s evaluation of a
time-dependent scene at every sample's own time, which remains the choice
when the scene moves within an observation.

[`Epochs`][virgil.epochs.Epochs] keys epochs and datasets **by name**,
so that per-dataset nuisance terms (error scales, wavelength scales,
North angles) are attached to the right data however the list is
ordered, and it hands [`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model] what they take: a
model function returning one snapshot per dataset, the datasets, and one
``noise`` dict per dataset.
"""

import numpy as onp

from .likelihood import model_loglike


__all__ = ["Epochs"]

_AT = ("dataset", "epoch")


class Epochs:
    """Datasets grouped into named epochs, each seen as one snapshot.

    Parameters
    ----------
    epochs : dict
        ``{epoch name: data}``, where ``data`` is one
        [`OIData`][virgil.oidata.OIData], a list of them, or a dict
        ``{dataset name: OIData}``. One epoch may hold several datasets,
        e.g. several files of one night, or two filters a day apart. A
        lone dataset is named after its epoch; datasets in a list are
        named ``"<epoch>[k]"``. Every name must be unique.
    at : {"dataset", "epoch"}, optional
        Where each snapshot is taken: at the mean time of each dataset's
        samples (default), or at the mean time of the whole epoch, shared
        by its datasets.
    times : dict, optional
        ``{epoch or dataset name: MJD}``: snapshot times to use instead of
        the data's own, e.g. for data without per-sample times (AMIGO
        DISCOs). A dataset's entry overrides its epoch's.

    Attributes
    ----------
    names : tuple of str
        The epochs, in the order given.
    dataset_names : tuple of str
        The datasets, epoch by epoch.
    epoch_of : tuple of str
        The epoch of each dataset.
    data : tuple of OIData
        The datasets, in the order of ``dataset_names``: pass this as
        ``data`` to ``fit`` and ``numpyro_model``.
    times : numpy.ndarray
        The snapshot time of each dataset (MJD, float64).
    spread_days : numpy.ndarray
        The largest distance in time of each dataset's samples from its
        snapshot (days). The scene should move by much less than the
        resolution (λ/B) over this time; if it does not, evaluate the
        dataset per sample instead (pass it to ``fit`` directly with the
        time-dependent scene).

    Examples
    --------
    >>> import numpy as np
    >>> from virgil.epochs import Epochs
    >>> from virgil.oidata import OIData
    >>> def night(mjd):
    ...     return OIData({"u": [10.0, 20.0], "v": [5.0, -3.0],
    ...                    "wavel": 2.2e-6, "vis": [1.0, 1.0],
    ...                    "d_vis": [0.01, 0.01], "mjd": [mjd, mjd + 0.1]})
    >>> epochs = Epochs({"2023": {"ut": night(60100.0), "at": night(60101.0)},
    ...                  "2024": night(60500.0)})
    >>> epochs.dataset_names, epochs.epoch_of
    (('ut', 'at', '2024'), ('2023', '2023', '2024'))
    >>> [round(float(t), 2) for t in epochs.times]
    [60100.05, 60101.05, 60500.05]
    >>> epochs.noise({"2023": {"vis_scale": 2.0}, "2024": {"vis_scale": 3.0}})
    [{'vis_scale': 2.0}, {'vis_scale': 2.0}, {'vis_scale': 3.0}]
    """

    def __init__(self, epochs, at="dataset", times=None):
        if at not in _AT:
            raise ValueError(f"at must be one of {_AT}, got {at!r}.")
        if not isinstance(epochs, dict) or not epochs:
            raise ValueError(
                "epochs must be a non-empty dict {epoch name: data}."
            )
        names, epoch_of, data = [], [], []
        for epoch, group in epochs.items():
            if isinstance(group, dict):
                items = list(group.items())
            elif isinstance(group, (list, tuple)):
                items = [(f"{epoch}[{k}]", d) for k, d in enumerate(group)]
            else:
                items = [(epoch, group)]
            if not items:
                raise ValueError(f"Epoch {epoch!r} has no data.")
            for name, d in items:
                names.append(name)
                epoch_of.append(epoch)
                data.append(d)
        duplicated = sorted(
            {n for n in names if names.count(n) > 1}
            | {n for n, e in zip(names, epoch_of) if n in epochs and n != e}
        )
        if duplicated:
            raise ValueError(
                f"Names must be unique; {duplicated} name both an epoch "
                "and a dataset, or two datasets."
            )
        self.names = tuple(epochs)
        self.dataset_names = tuple(names)
        self.epoch_of = tuple(epoch_of)
        self.data = tuple(data)
        self.at = at
        self.times, self.spread_days = self._snapshot_times(times or {})

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return (
            f"Epochs({len(self.names)} epochs, {len(self.data)} datasets, "
            f"at={self.at!r})"
        )

    def _snapshot_times(self, given):
        """Snapshot time and spread of each dataset."""
        unknown = set(given) - set(self.names) - set(self.dataset_names)
        if unknown:
            raise ValueError(f"times for unknown names: {sorted(unknown)}.")
        own = []
        for name, d in zip(self.dataset_names, self.data):
            mjd = getattr(d, "mjd", None)
            own.append(None if mjd is None else onp.asarray(mjd, onp.float64))
        times = onp.empty(len(self.data))
        for k, (name, epoch) in enumerate(
            zip(self.dataset_names, self.epoch_of)
        ):
            if name in given:
                times[k] = given[name]
            elif epoch in given:
                times[k] = given[epoch]
            elif own[k] is None:
                raise ValueError(
                    f"Dataset {name!r} has no times: read it from OIFITS, "
                    "give mjd per sample, or pass times={name: mjd}."
                )
            elif self.at == "dataset":
                times[k] = own[k].mean()
            else:
                together = [
                    own[j]
                    for j, e in enumerate(self.epoch_of)
                    if e == epoch and own[j] is not None
                ]
                times[k] = onp.concatenate(together).mean()
        spread = onp.array(
            [
                0.0 if t is None else float(onp.max(onp.abs(t - times[k])))
                for k, t in enumerate(own)
            ]
        )
        return times, spread

    def index(self, name):
        """The position of dataset ``name`` in ``data`` (and in ``noise``).

        Fitted per-dataset noise terms are the sites
        ``"noise[<index>].<term>"`` of ``fit`` and ``numpyro_model``.
        """
        try:
            return self.dataset_names.index(name)
        except ValueError:
            raise KeyError(
                f"No dataset {name!r}; the datasets are "
                f"{list(self.dataset_names)}."
            ) from None

    def snapshots(self, scene):
        """The scene at each dataset's snapshot time, one model per dataset.

        Parameters
        ----------
        scene : SourceModel or sequence of SourceModel
            One scene for all datasets, or one per dataset (e.g. with
            per-dataset fluxes). A static scene is its own snapshot.

        Returns
        -------
        list of SourceModel
            Static models, in the order of ``data``.
        """
        scenes = (
            list(scene)
            if isinstance(scene, (list, tuple))
            else [scene] * len(self.data)
        )
        if len(scenes) != len(self.data):
            raise ValueError(
                f"{len(scenes)} scenes for {len(self.data)} datasets."
            )
        return [s.at(float(t)) for s, t in zip(scenes, self.times)]

    def model_fn(self, scene_fn):
        """A model function for ``fit``/``numpyro_model``: the snapshots of
        ``scene_fn(**values)``, one per dataset."""

        def snapshots(**values):
            return self.snapshots(scene_fn(**values))

        return snapshots

    def noise(self, terms):
        """One ``noise`` dict per dataset, from terms keyed by name.

        Parameters
        ----------
        terms : dict
            ``{epoch or dataset name: {term: prior, value or tied
            callable}}``. An epoch's terms apply to each of its datasets,
            and each dataset gets its own copy (its own fitted site); a
            dataset's own terms are added to, and override, its epoch's.
            To share one fitted value between datasets, tie them to one
            parameter with callables (see
            [`noise_sites`][virgil.likelihood.noise_sites]).

        Returns
        -------
        list of dict
            In the order of ``data``, as ``fit(..., noise=...)`` takes.
        """
        unknown = set(terms) - set(self.names) - set(self.dataset_names)
        if unknown:
            raise KeyError(
                f"noise for unknown names {sorted(unknown)}; the epochs are "
                f"{list(self.names)} and the datasets "
                f"{list(self.dataset_names)}."
            )
        return [
            {**terms.get(epoch, {}), **terms.get(name, {})}
            for name, epoch in zip(self.dataset_names, self.epoch_of)
        ]

    def loglike(self, scene, noise=None):
        """The log likelihood of all the data, one snapshot per dataset.

        The sum over datasets of
        [`model_loglike`][virgil.likelihood.model_loglike] of each
        snapshot, with ``noise`` (values, keyed by name as for
        :meth:`noise`) applied to each dataset. Traceable: it can be
        jitted, differentiated or vmapped in the scene's parameters.
        """
        per_dataset = self.noise(noise or {})
        return sum(
            model_loglike(snap, d, **n)
            for snap, d, n in zip(
                self.snapshots(scene), self.data, per_dataset
            )
        )
