"""Symbolic regression layer built on HFF.

The engine and its operators grew up as loose modules in ``notebooks/``. This
package is where they live as real library code. Modules are added here one at
a time; the old ``notebooks/`` files re-export from here, so both import paths
work and nothing has to migrate in a big bang.

Public surface so far:
    simplify_kexpression_bounded() — bounded replacement for geppy's
        _simplify_kexpression, which calls sp.simplify at every internal node
        and goes exponential with gene depth.
"""

from hff.sr.bounded_simplify import (
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
