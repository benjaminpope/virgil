from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as np

from virgil.amigo import load_oi_data
from virgil.likelihood import model_loglike
from virgil.models import BinaryModelCartesian


DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "calibrated_visibility.npy"
)


@dataclass(frozen=True)
class AmigoDiscoSummary:
    filters: tuple[str, ...]
    single_filter: str
    n_observables: int
    single_filter_loglike: float
    joint_loglike: float


def summarize_amigo_disco_product(
    path: str | Path = DATA_PATH,
) -> AmigoDiscoSummary:
    """Load an AMIGO mixed-DISCO product and evaluate a simple binary model."""
    observations = load_oi_data(path)
    filters = tuple(observations)

    single_filter = "F430M" if "F430M" in observations else filters[0]
    model = BinaryModelCartesian(dra=100.0, ddec=-50.0, flux=1e-3)
    single_filter_loglike = float(
        model_loglike(model, observations[single_filter])
    )

    flux_by_filter = {
        "F380M": 8e-4,
        "F430M": 1e-3,
        "F480M": 1.2e-3,
    }
    joint_loglike = sum(
        model_loglike(
            BinaryModelCartesian(
                dra=100.0,
                ddec=-50.0,
                flux=flux_by_filter.get(filter_name, 1e-3),
            ),
            oidata,
        )
        for filter_name, oidata in observations.items()
    )

    return AmigoDiscoSummary(
        filters=filters,
        single_filter=single_filter,
        n_observables=int(np.asarray(observations[single_filter].vis).size),
        single_filter_loglike=single_filter_loglike,
        joint_loglike=float(joint_loglike),
    )


def main() -> None:
    summary = summarize_amigo_disco_product()
    print(f"filters: {', '.join(summary.filters)}")
    print(
        f"{summary.single_filter}: {summary.n_observables} mixed-DISCO observables"
    )
    print(
        f"{summary.single_filter} log likelihood: {summary.single_filter_loglike:.6g}"
    )
    print(f"joint log likelihood: {summary.joint_loglike:.6g}")


if __name__ == "__main__":
    main()
