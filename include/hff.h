/*
 * hff.h -- C ABI for the HFF (Hyperspherical Fitness Functions) Rust library.
 *
 * Built by: cargo build --release --features c-api
 * Produces: libhff_core.{dylib, so, dll}
 *
 * All functions are pure and thread-safe (Rayon handles internal parallelism).
 * Callers own all buffers; no allocations cross the FFI boundary.
 *
 * Matrices are ROW-MAJOR: element (i, j) at index i * n_cols + j.
 */

#ifndef HFF_H
#define HFF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
#define HFF_OK            0
#define HFF_ERR_NULL     -1
#define HFF_ERR_INVALID  -2
#define HFF_ERR_INTERNAL -3

/*
 * HF1 plain (f64) -- caller is responsible for any normalisation.
 *
 *   objectives      row-major, length = n_individuals * n_objectives
 *   decrowding      0 or 1 (apply softplus(z-score) decrowding)
 *   out_fitness     length = n_individuals; angular distances in radians [0, pi]
 */
int32_t hff_hf1_f64(
    const double* objectives,
    size_t        n_individuals,
    size_t        n_objectives,
    int32_t       decrowding,
    double*       out_fitness
);

/*
 * HF1 enhanced -- performs column-wise min-max normalisation internally and
 * supports the two north-pole methods.
 *
 *   north_pole_method   null-terminated C string, "balanced" or "truenorth".
 *                       Pass NULL for "balanced".
 */
int32_t hff_hf1_enhanced(
    const double* objectives,
    size_t        n_individuals,
    size_t        n_objectives,
    int32_t       decrowding,
    const char*   north_pole_method,
    double*       out_fitness
);

/*
 * HIGD -- CDF-corrected angular IGD; set-level quality metric.
 *
 *   solutions           row-major, length = n_solutions * dimensions
 *   positive_orthant    0 (full sphere) or 1 (positive orthant, for WFG4-9)
 *   out_value           written with result in [0, 1]
 */
int32_t hff_higd(
    const double* solutions,
    size_t        n_solutions,
    size_t        dimensions,
    size_t        n_reference_points,
    uint64_t      seed,
    int32_t       positive_orthant,
    double*       out_value
);

/*
 * CDF-correct a raw angular distance theta (radians) given objective count.
 * Returns dimension-independent percentile in [0, 1] — makes fitness values
 * comparable across runs with different numbers of objectives.
 *
 * For large `dimensions` and small theta the result underflows f64 — in that
 * regime use hff_log_cdf_correction instead.
 */
double hff_cdf_correction(double theta, size_t dimensions);

/*
 * Log of the Beta-CDF correction. Always representable in f64 (no underflow).
 * Preferred when many-objective fitness values live in the deep left tail.
 */
double hff_log_cdf_correction(double theta, size_t dimensions);

/*
 * Raw angular IGD (no CDF correction). Same argument semantics as hff_higd;
 * out_value is the mean minimum angular distance in radians.
 */
int32_t hff_angular_igd(
    const double* solutions,
    size_t        n_solutions,
    size_t        dimensions,
    size_t        n_reference_points,
    uint64_t      seed,
    int32_t       positive_orthant,
    double*       out_value
);

#ifdef __cplusplus
}
#endif

#endif /* HFF_H */
