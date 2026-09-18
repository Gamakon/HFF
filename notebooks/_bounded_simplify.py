"""Compatibility shim — the implementation now lives in ``hff.sr``.

Kept so existing ``from _bounded_simplify import ...`` sites in notebooks/
keep working while code migrates to the package. New code should import from
``hff.sr`` directly::

    from hff.sr import simplify_kexpression_bounded

The module-level stats are the *same objects* as the package's, so
``get_stats()`` reports identically whichever path a caller used.
"""

from hff.sr.bounded_simplify import (  # noqa: F401
    simplify_kexpression_bounded,
    get_stats,
    reset_stats,
    DEFAULT_MAX_NODES,
)

__all__ = [
    "simplify_kexpression_bounded",
    "get_stats",
    "reset_stats",
    "DEFAULT_MAX_NODES",
]
