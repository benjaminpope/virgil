import numpy as onp
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from tests.test_orbit_cells import _summaries, _state, LOG_B
from tests.test_aliases import EPOCHS, T_REF, TRUTH
from virgil.orbits import KeplerOrbit
from virgil import orbit_cells as oc

for sigma in (0.3, 0.1, 0.03, 0.01):
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    s = _summaries(orbit, EPOCHS, sigma=sigma)
    cat = oc._Catalogue(s, onp.full(6, LOG_B)).cast(np.float64)
    pv = np.asarray(_state(orbit, onp.asarray(EPOCHS, float)))
    ln_s = np.zeros(len(cat.blocks))
    lt = oc._log_mixture(cat, ln_s, 0.0, pv, -2.0, np.float64)
    lfar = oc._log_mixture(cat, ln_s, 0.0, pv * 0 + 5.0, -2.0, np.float64)
    # perturb period by small amounts: how wide is the inlier region in P
    out = []
    for dP in (1e-4, 1e-3, 1e-2):
        o2 = KeplerOrbit(
            **{**TRUTH, "period": TRUTH["period"] + dP}, t_ref=T_REF
        )
        out.append(
            float(
                oc._log_mixture(
                    cat,
                    ln_s,
                    0.0,
                    np.asarray(_state(o2, onp.asarray(EPOCHS, float))),
                    -2.0,
                    np.float64,
                )
            )
        )
    print(
        f"sigma={sigma}: lnL(truth)={float(lt):.2f} floor={float(lfar):.2f} gap={float(lt - lfar):.1f}  lnL(P+1e-4,1e-3,1e-2, other params fixed)={[round(x, 1) for x in out]}"
    )
print(TRUTH)
