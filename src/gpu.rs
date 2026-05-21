//! GPU-accelerated HFF batch fitness via wgpu compute shader.
//!
//! Same math as `core_functions::calculate_single_hyperspherical_fitness_f64_with_method`
//! (with the same column-wise min-max normalisation as the CPU batch entry
//! point in `lib.rs`), but executed on the GPU with one thread per row.
//!
//! Architecture decisions (cribbed from qdrant/hffvenn patterns):
//!   - Column-wise min/max is computed on the CPU and passed as a uniform.
//!     Cheap (O(N·M) on the host) and avoids a separate reduction pass.
//!   - One thread per row, workgroup_size = 64. Dispatches `ceil(N/64)`.
//!   - f32 precision on the GPU. Caller passes f32; we cast back to f64 for
//!     Python API parity. The CPU path is f64; expected discrepancy ≤ 1e-6.
//!   - Synchronous device init via `pollster::block_on`. One device per
//!     call today (no shared device manager yet — add when we plumb it
//!     into multi-call workloads).
//!
//! Limitations:
//!   - North pole method is "truenorth" only (the documented config for the
//!     paper, and what every notebook uses). "balanced" path can be added
//!     by overloading the shader entry point.
//!   - No tiling yet. If N·M·4 exceeds the per-binding storage limit
//!     (~128 MB on Apple Metal), the call errors out. Add candidate-axis
//!     tiling later (qdrant §8.5.1 pattern).

use bytemuck::{Pod, Zeroable};
use ndarray::Array2;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct Uniforms {
    n_individuals: u32,
    n_objectives: u32,
    _pad0: u32,
    _pad1: u32,
}

const WGSL_SHADER: &str = r#"
struct Uniforms {
    n_individuals: u32,
    n_objectives: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// objectives_normalized: row-major [n_individuals × n_objectives], f32.
// Already min-max normalised on the host so values in [0, 1].
@group(0) @binding(1) var<storage, read> objectives_normalized: array<f32>;
// distances out: array<f32>, length n_individuals.
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn hf1_truenorth_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= uniforms.n_individuals) {
        return;
    }

    let m = uniforms.n_objectives;
    let row_base = row * m;

    // The CPU pre-pass replaces any non-finite input with the sentinel
    // 1e30 (arbitrarily large but finite). The shader bails out on any
    // value >= 1e29. We don't rely on WGSL NaN/Inf semantics (which are
    // implementation-defined across backends).
    var energy_sum: f32 = 0.0;
    var any_bad: bool = false;
    for (var j: u32 = 0u; j < m; j = j + 1u) {
        let v = objectives_normalized[row_base + j];
        if (v >= 1.0e29 || v <= -1.0e29) { any_bad = true; }
        energy_sum = energy_sum + v * v;
    }
    if (any_bad) {
        distances[row] = 3.14159265358979;
        return;
    }

    if (energy_sum <= 1.0e-12) {
        distances[row] = 0.0;
        return;
    }

    // TrueNorth: energy_score in [0, 1], dot with pole = energy_score.
    let m_f = f32(m);
    let normalized_energy = min(energy_sum / m_f, 1.0);
    let energy_score = max(1.0 - normalized_energy, 0.0);

    // cos_theta = energy_score (because dot product reduces to that under
    // the augmented projection — see core_functions.rs comment block).
    var cos_theta = energy_score;
    cos_theta = clamp(cos_theta, -1.0, 1.0);

    var theta: f32;
    if (abs(cos_theta) > 1.0 - 1.0e-7) {
        if (cos_theta > 0.0) {
            theta = 0.0;
        } else {
            theta = 3.14159265358979;
        }
    } else {
        theta = acos(cos_theta);
    }

    if (theta != theta || theta > 3.4e38 || theta < -3.4e38) {
        distances[row] = 3.14159265358979;
    } else {
        distances[row] = theta;
    }
}
"#;

/// GPU-side device + pipeline. Built once, can be reused across calls.
pub struct HffGpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl HffGpuContext {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "no GPU adapter available".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("HffGpuDevice"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("device request failed: {e:?}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HffGpuShader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HffGpuBindGroupLayout"),
            entries: &[
                // 0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: normalized objectives (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: distances out (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HffGpuPipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HffGpuPipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "hf1_truenorth_main",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Run HF1 truenorth on a row-major [N × M] f64 matrix. The CPU does
    /// column min-max normalisation and casts to f32; GPU produces f32
    /// distances; caller gets f64 for API parity.
    ///
    /// Robustness contract:
    ///   - n == 0 → empty Vec (Ok).
    ///   - m == 0 → Err.
    ///   - n*m*4 over per-binding storage limit → Err with explanation.
    ///   - NaN/Inf in any input row → that row gets PI (max angular distance).
    ///   - Constant columns (min==max) → mapped to 0 via range=1 fallback.
    ///   - Single individual (n=1) → normalization skipped (matches CPU).
    ///   - GPU device-lost during map → Err.
    pub fn calculate_hf1_truenorth_batch(
        &self,
        objectives: &Array2<f64>,
        normalize: bool,
    ) -> Result<Vec<f64>, String> {
        let (n, m) = objectives.dim();
        if n == 0 {
            return Ok(Vec::new());
        }
        if m == 0 {
            return Err("n_objectives must be > 0".to_string());
        }

        // Pre-check buffer size against the adapter's per-binding limit. The
        // objectives buffer is n*m*4 bytes; if it exceeds the cap we error
        // BEFORE allocating to give a clear message.
        let obj_bytes = (n as u64).saturating_mul(m as u64).saturating_mul(4);
        let dist_bytes = (n as u64).saturating_mul(4);
        let max_storage = self.device.limits().max_storage_buffer_binding_size as u64;
        if obj_bytes > max_storage {
            return Err(format!(
                "objectives buffer {}B exceeds adapter limit {}B (n={}, m={}). \
                 Tile the input across the candidate axis or use CPU path.",
                obj_bytes, max_storage, n, m
            ));
        }
        if dist_bytes > max_storage {
            return Err(format!(
                "distances buffer {}B exceeds adapter limit {}B (n={}). \
                 Tile across candidate axis.",
                dist_bytes, max_storage, n
            ));
        }

        // CPU: column-wise min-max normalise (matches lib.rs:227-258).
        // Robust to NaN/Inf: those values propagate to the GPU and the
        // shader's any_bad branch returns PI per affected row.
        let normalized_f32: Vec<f32> = if normalize && n > 1 {
            let mut buf = vec![0.0f32; n * m];
            for j in 0..m {
                let mut col_min = f64::INFINITY;
                let mut col_max = f64::NEG_INFINITY;
                for i in 0..n {
                    let v = objectives[[i, j]];
                    // Skip NaN when computing min/max; they'd corrupt the
                    // range. The NaN itself still flows through to the GPU
                    // and triggers the any_bad branch.
                    if v.is_finite() {
                        if v < col_min {
                            col_min = v;
                        }
                        if v > col_max {
                            col_max = v;
                        }
                    }
                }
                let range = if !col_min.is_finite() || !col_max.is_finite()
                    || (col_max - col_min) < f64::EPSILON
                {
                    1.0
                } else {
                    col_max - col_min
                };
                let safe_min = if col_min.is_finite() { col_min } else { 0.0 };
                for i in 0..n {
                    let v = objectives[[i, j]];
                    // Sentinel 1e30 marks non-finite rows for the shader.
                    // WGSL NaN/Inf semantics are implementation-defined, so
                    // we use a finite sentinel that the shader recognises.
                    buf[i * m + j] = if v.is_finite() {
                        ((v - safe_min) / range) as f32
                    } else {
                        1.0e30f32
                    };
                }
            }
            buf
        } else {
            let mut buf = vec![0.0f32; n * m];
            for i in 0..n {
                for j in 0..m {
                    let v = objectives[[i, j]];
                    buf[i * m + j] = if v.is_finite() { v as f32 } else { 1.0e30f32 };
                }
            }
            buf
        };

        let uniforms = Uniforms {
            n_individuals: n as u32,
            n_objectives: m as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let uniforms_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("HffGpuUniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let obj_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("HffGpuObjectives"),
                contents: bytemuck::cast_slice(&normalized_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let dist_buf_size = (n * std::mem::size_of::<f32>()) as u64;
        let dist_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HffGpuDistances"),
            size: dist_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HffGpuStaging"),
            size: dist_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HffGpuBindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: obj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dist_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("HffGpuEncoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HffGpuComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let n_workgroups = ((n as u32) + 63) / 64;
            cpass.dispatch_workgroups(n_workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dist_buf, 0, &staging_buf, 0, dist_buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back. The map_async callback fires once the queue completes
        // the copy. Both the channel-disconnect case and the wgpu-error case
        // are surfaced as Err so the caller can recover (instead of panic).
        let buffer_slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            // If the receiver was dropped (which shouldn't happen here since
            // rx lives until after recv()), silently drop the result —
            // we'd then surface the timeout/disconnect as a recv error.
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let map_result = rx.recv().map_err(|e| {
            format!("GPU map channel disconnected (device lost?): {e:?}")
        })?;
        map_result.map_err(|e| format!("GPU buffer map failed: {e:?}"))?;
        let data = buffer_slice.get_mapped_range();
        let f32_results: &[f32] = bytemuck::cast_slice(&data);
        // Coerce any residual NaN to PI (defensive: shader should have
        // already done this, but f32 round-trip can produce subnormals).
        let out: Vec<f64> = f32_results
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    v as f64
                } else {
                    std::f64::consts::PI
                }
            })
            .collect();
        drop(data);
        staging_buf.unmap();

        Ok(out)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_functions::calculate_single_hyperspherical_fitness_f64_with_method;
    use ndarray::{Array1, Array2};

    /// Run the same row through the CPU path so we can compare. Mirrors
    /// lib.rs's enhanced entry point (incl. min-max normalize).
    fn cpu_reference(f: &Array2<f64>, normalize: bool) -> Vec<f64> {
        let (n, m) = f.dim();
        let normed = if normalize && n > 1 {
            let mut out = f.clone();
            for j in 0..m {
                let col: Vec<f64> = (0..n).map(|i| f[[i, j]]).collect();
                let lo = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let hi = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let range = if (hi - lo) < f64::EPSILON { 1.0 } else { hi - lo };
                for i in 0..n {
                    out[[i, j]] = (f[[i, j]] - lo) / range;
                }
            }
            out
        } else {
            f.clone()
        };
        (0..n)
            .map(|i| {
                let row: Array1<f64> = normed.slice(ndarray::s![i, ..]).to_owned();
                calculate_single_hyperspherical_fitness_f64_with_method(
                    &row,
                    m,
                    false,
                    None,
                    "truenorth",
                )
            })
            .collect()
    }

    #[test]
    fn parity_basic_500x6() {
        let ctx = HffGpuContext::new().expect("GPU init");
        let n = 500;
        let m = 6;
        let mut data = Vec::with_capacity(n * m);
        let mut seed = 0u64;
        for _ in 0..n * m {
            // Simple xorshift for reproducibility without a crate dep.
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            data.push((seed % 1_000_000) as f64 / 100_000.0);
        }
        let f = Array2::from_shape_vec((n, m), data).unwrap();
        let gpu = ctx.calculate_hf1_truenorth_batch(&f, true).unwrap();
        let cpu = cpu_reference(&f, true);
        for i in 0..n {
            let diff = (gpu[i] - cpu[i]).abs();
            assert!(diff < 1e-5, "row {i}: cpu={} gpu={} diff={diff}", cpu[i], gpu[i]);
        }
    }

    #[test]
    fn empty_input() {
        let ctx = HffGpuContext::new().expect("GPU init");
        let f = Array2::<f64>::zeros((0, 6));
        let out = ctx.calculate_hf1_truenorth_batch(&f, true).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_objectives_errors() {
        let ctx = HffGpuContext::new().expect("GPU init");
        let f = Array2::<f64>::zeros((10, 0));
        let result = ctx.calculate_hf1_truenorth_batch(&f, true);
        assert!(result.is_err());
    }

    #[test]
    fn single_individual_passes() {
        // n=1 → normalize is skipped (matches CPU). The lone row should
        // produce a finite angular distance, not a panic.
        let ctx = HffGpuContext::new().expect("GPU init");
        let f = Array2::from_shape_vec((1, 6), vec![0.5, 0.3, 0.8, 0.1, 0.2, 0.4]).unwrap();
        let out = ctx.calculate_hf1_truenorth_batch(&f, true).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite());
        assert!((0.0..=std::f64::consts::PI).contains(&out[0]));
    }

    #[test]
    fn constant_column_handled() {
        // One column has every row = 5.0. After min-max normalize that column
        // becomes 0 everywhere (range=1 fallback). Should produce finite
        // results, no NaN/Inf.
        let ctx = HffGpuContext::new().expect("GPU init");
        let n = 50;
        let m = 6;
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            data[i * m + 0] = 5.0;             // constant
            data[i * m + 1] = i as f64 * 0.1;  // varying
            data[i * m + 2] = (i * 2) as f64 * 0.05;
            data[i * m + 3] = 5.0;             // constant
            data[i * m + 4] = (n - i) as f64 * 0.07;
            data[i * m + 5] = ((i * 7) % 13) as f64 * 0.03;
        }
        let f = Array2::from_shape_vec((n, m), data).unwrap();
        let out = ctx.calculate_hf1_truenorth_batch(&f, true).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "row {i}: {v}");
        }
    }

    #[test]
    fn nan_input_becomes_pi() {
        // A row containing NaN must come back as PI (max angular distance),
        // not panic, not corrupt other rows.
        let ctx = HffGpuContext::new().expect("GPU init");
        let mut data = vec![0.5; 5 * 6];
        data[2 * 6 + 3] = f64::NAN;
        let f = Array2::from_shape_vec((5, 6), data).unwrap();
        let out = ctx.calculate_hf1_truenorth_batch(&f, true).unwrap();
        assert_eq!(out.len(), 5);
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "row {i}: {v}");
            if i == 2 {
                assert!(
                    (v - std::f64::consts::PI).abs() < 1e-4,
                    "row 2 should be PI, got {v}"
                );
            }
        }
    }

    #[test]
    fn inf_input_becomes_pi() {
        let ctx = HffGpuContext::new().expect("GPU init");
        let mut data = vec![0.5; 5 * 6];
        data[1 * 6 + 0] = f64::INFINITY;
        data[3 * 6 + 2] = f64::NEG_INFINITY;
        let f = Array2::from_shape_vec((5, 6), data).unwrap();
        let out = ctx.calculate_hf1_truenorth_batch(&f, true).unwrap();
        assert_eq!(out.len(), 5);
        for v in out.iter() {
            assert!(v.is_finite(), "got {v}");
        }
        // The Inf rows should be PI.
        assert!((out[1] - std::f64::consts::PI).abs() < 1e-4);
        assert!((out[3] - std::f64::consts::PI).abs() < 1e-4);
    }

    #[test]
    fn all_zeros_gives_zero_distance() {
        // Zero energy → perfect minimisation → angular distance 0.
        let ctx = HffGpuContext::new().expect("GPU init");
        let f = Array2::<f64>::zeros((10, 6));
        let out = ctx.calculate_hf1_truenorth_batch(&f, false).unwrap();
        for v in out.iter() {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn dimensions_extreme_m_passes() {
        // High-objective case (m=100). Tests the inner loop scaling.
        let ctx = HffGpuContext::new().expect("GPU init");
        let n = 64;
        let m = 100;
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i * m + j] = ((i + 1) as f64 / (j + 1) as f64).sin().abs();
            }
        }
        let f = Array2::from_shape_vec((n, m), data).unwrap();
        let out = ctx.calculate_hf1_truenorth_batch(&f, true).unwrap();
        let cpu = cpu_reference(&f, true);
        assert_eq!(out.len(), cpu.len());
        for i in 0..n {
            let diff = (out[i] - cpu[i]).abs();
            assert!(diff < 1e-4, "row {i}: cpu={} gpu={} diff={diff}", cpu[i], out[i]);
        }
    }

    #[test]
    fn dispatch_boundary_workgroup_align() {
        // n = workgroup_size + 1 → tests that the last workgroup's spillover
        // thread correctly bails (gid.x >= n_individuals).
        let ctx = HffGpuContext::new().expect("GPU init");
        let n = 65;
        let m = 6;
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i * m + j] = ((i + j) as f64) * 0.013;
            }
        }
        let f = Array2::from_shape_vec((n, m), data).unwrap();
        let out = ctx.calculate_hf1_truenorth_batch(&f, true).unwrap();
        assert_eq!(out.len(), n);
        for v in out.iter() {
            assert!(v.is_finite() && (0.0..=std::f64::consts::PI).contains(v));
        }
    }
}
