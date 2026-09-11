"""Restraint-Guided Inference toolkit.

The top-level package imports only numpy-level modules so that ``import rgi_toolkit``
works without torch or jax installed. Heavy backends (torch/jax) are imported
lazily by ``combined.py`` / the ``energy`` and ``optim`` subpackages.
"""

import logging

from rgi_toolkit.atom_context import (
    AtomRecord,
    ConformerAdapter,
    FrameworkAdapter,
    LigandConf,
)
from rgi_toolkit.combined import CombinedRestraints
from rgi_toolkit.config import resolve_restraints_config
from rgi_toolkit.custom import custom_restraint
from rgi_toolkit.optim.info import CGInfo, CGStatus

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AtomRecord",
    "FrameworkAdapter",
    "ConformerAdapter",
    "LigandConf",
    "CombinedRestraints",
    "resolve_restraints_config",
    "CGInfo",
    "CGStatus",
    # custom restraints: register a reusable code energy fn (config refs it by {use: name})
    "custom_restraint",
]
