The all-outlier floor check in `orbit_cells.position_leg` now runs before the search, so a non-finite outlier density fails at once instead of after a full cell search (the error is unchanged).
