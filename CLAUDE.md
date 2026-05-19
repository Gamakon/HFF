# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & test

This is a Rust crate (`hff_core`) with two optional binding layers, gated by Cargo features:
- `python` (default) — PyO3 module, built via maturin into `python/hff/hff_core.*.so`
- `c-api` — `extern "C"` symbols exported from the `cdylib`; header at `include/hff.h`

```bash
# Rust-only tests (no Python feature; runs higd.rs unit tests)
cargo test --no-default-features

# Rust tests with the C ABI (runs c_api.rs tests)
cargo test --no-default-features --features c-api

# A single test
cargo test --no-default-features test_higd_range

# Python build + install into the active venv
maturin develop --release

# C shared library
cargo build --release --features c-api
# Produces target/release/libhff_core.{dylib,so,dll}
```

Note: `cargo test` without `--no-default-features` activates `python`, which requires the PyO3 link environment and usually fails outside `maturin`. Always pass `--no-default-features` for plain Rust testing.

## Architecture

The crate is structured as a **pure-Rust core** with **two thin binding shims** layered on top. Both shims call the same `core_functions` / `higd` modules; keep them numerically in lockstep.

```
src/core_functions.rs    Single-individual HF1 math (f64). Pure, no FFI.
src/higd.rs              HIGD set-level metric + Beta-CDF correction. Pure, no FFI.
src/lib.rs               PyO3 wrappers (#[cfg(feature = "python")])
src/c_api.rs             extern "C" wrappers (#[cfg(feature = "c-api")])
python/hff/core.py       Python convenience layer (normalization, ndim handling)
python/hff/__init__.py   Re-exports the public API
include/hff.h            C header — must stay in sync with c_api.rs signatures
```

### Algorithm surface

Four operations, exposed identically from all three frontends (Rust, Python, C):

1. **HF1 Balanced** — angular distance from a solution to the diagonal pole `(1/√m, …, 1/√m)` on the unit hypersphere. Solution-level fitness.
2. **HF1 TrueNorth** — augmented-space variant: builds a point in ℝ^(m+1) whose extra coordinate encodes the solution's energy, with the reference pole at `(0,…,0,1)`. Use for direct minimization rather than balanced trade-offs.
3. **HIGD** — set-level quality indicator. For each of N uniform reference points on the sphere, find the nearest solution's angular distance, then apply the Beta-CDF correction `I_{sin²θ}((d−1)/2, 1/2)` so values are comparable across dimensions.
4. **Angular IGD** — same as HIGD but without CDF correction; returns raw mean angular distance in radians.

### Normalization rule (critical)

`hff_hf1_f64` / `calculate_hyperspherical_fitness_hf1_f64` assume the input is **already** column-wise min-max normalized. The "enhanced" variants (and `python/hff/core.py`'s `calculate_fitness_hf1`) normalize internally. The Python wrapper and the C `hff_hf1_enhanced` both implement the same min-max normalization — if you change one, change the other and verify the C test `hf1_enhanced_matches_plain_on_prenormalized_input` still passes.

### CDF correction & the log-space path

`higd::cdf_beta_correction` underflows f64 for large `dimensions` and small `theta` (the typical regime for image-reconstruction-style problems with hundreds of objectives). For those cases use `log_cdf_beta_correction` / `hff_log_cdf_correction`, which keeps the prefactor in log space and evaluates Lentz's continued fraction in linear space. Anything in the deep left tail must use the log variant — the linear one returns zero there.

### Parallelism

Hot loops over individuals / reference points use Rayon (`into_par_iter`). The library has no global state and is thread-safe; callers do not need to manage the pool.

### When adding a function

It will typically need to land in four places to stay coherent:
1. The pure-Rust implementation in `core_functions.rs` or `higd.rs`
2. The PyO3 wrapper in `src/lib.rs` (and register it in the `hff_core` `#[pymodule]`)
3. The C ABI wrapper in `src/c_api.rs` (with `#[no_mangle]` + safety doc + null-pointer guards returning `HFF_ERR_NULL`)
4. The C declaration in `include/hff.h`

Optionally also `python/hff/core.py` if a numpy-friendly convenience wrapper is wanted, and re-export from `python/hff/__init__.py`.
