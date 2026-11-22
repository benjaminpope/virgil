Split `tests/test_orbit_cells.py` into three files (pure moves) and shard the full tiers into 9 float32 and 6 float64 jobs, so no `full-ci` shard is estimated above about 25 of its 45 minutes.
