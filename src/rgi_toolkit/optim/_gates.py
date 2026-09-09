"""Host-spec window tables shared by optimizer entry points."""

import numpy as np

from rgi_toolkit.energy._terms import PER_ENTRY_KEYS, iter_spec_terms


def active_windows(spec):
    """Collect real restraint windows without reading prepared device tensors."""
    rows = []
    if spec.has_conformer():
        rows.append(
            [
                spec.conf_start_sigma,
                spec.conf_stop_sigma,
                spec.conf_start_step,
                spec.conf_stop_step,
            ]
        )
    for _term, array in iter_spec_terms(spec, PER_ENTRY_KEYS):
        on = (np.asarray(array.mask) > 0) & (np.asarray(array.weight) != 0)
        rows.extend(
            np.stack(
                [
                    array.start_sigma,
                    array.stop_sigma,
                    array.start_step,
                    array.stop_step,
                ],
                axis=-1,
            )[on]
        )
    rows.extend(
        [c.start_sigma, c.stop_sigma, c.start_step, c.stop_step]
        for c in spec.custom
        if c.weight != 0
    )
    return np.asarray(rows, dtype=float).reshape((-1, 4))


def window_on(windows, sigma, step, xp=np):
    """Whether any window is active; omitted axes bypass their gate on Torch."""
    on = xp.ones(windows.shape[0], dtype=bool)
    if sigma is not None:
        on = on & (sigma <= windows[:, 0]) & (sigma >= windows[:, 1])
    if step is not None:
        on = on & (step >= windows[:, 2]) & (step <= windows[:, 3])
    return xp.any(on)
