#!/usr/bin/env bash
# Install HFF-SR for the SRBench harness.
# Clones the HFF repo, builds the Rust core via maturin, and exposes the
# notebook engine module on the Python path.

set -euo pipefail

# Where to drop the HFF source. Override HFF_INSTALL_PREFIX if you want
# a different location.
PREFIX="${HFF_INSTALL_PREFIX:-${CONDA_PREFIX:-$HOME/.local}}"
mkdir -p "$PREFIX"

if [ ! -d "$PREFIX/hff" ]; then
  git clone https://github.com/Gamakon/HFF.git "$PREFIX/hff"
fi

# Build the Rust core (default 'python' feature) and install into the
# active Python environment.
(cd "$PREFIX/hff" && maturin develop --release)

# Drop the engine module onto the Python path so ``import hff_sr_engine``
# works without referencing the notebooks/ directory.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp "$PREFIX/hff/notebooks/hff_sr_engine.py" "$SITE/"
cp "$PREFIX/hff/notebooks/hff_geppy_helpers.py" "$SITE/"
cp "$PREFIX/hff/notebooks/equation_problems.py" "$SITE/" 2>/dev/null || true

echo "HFF-SR install complete (prefix=$PREFIX, site=$SITE)"
