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

    // NaN/Inf check: caller is responsible. We bail out via NaN comparison
    // (NaN != NaN) and Inf magnitude.
    var energy_sum: f32 = 0.0;
    var any_bad: bool = false;
    for (var j: u32 = 0u; j < m; j = j + 1u) {
        let v = objectives_normalized[row_base + j];
        if (v != v) { any_bad = true; }                 // NaN
        if (v > 3.4e38 || v < -3.4e38) { any_bad = true; }  // ±Inf approx
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

        // CPU: column-wise min-max normalise (matches lib.rs:227-258).
        let normalized_f32: Vec<f32> = if normalize && n > 1 {
            let mut buf = vec![0.0f32; n * m];
            for j in 0..m {
                let mut col_min = f64::INFINITY;
                let mut col_max = f64::NEG_INFINITY;
                for i in 0..n {
                    let v = objectives[[i, j]];
                    if v < col_min {
                        col_min = v;
                    }
                    if v > col_max {
                        col_max = v;
                    }
                }
                let range = if (col_max - col_min) < f64::EPSILON {
                    1.0
                } else {
                    col_max - col_min
                };
                for i in 0..n {
                    buf[i * m + j] = ((objectives[[i, j]] - col_min) / range) as f32;
                }
            }
            buf
        } else {
            let mut buf = vec![0.0f32; n * m];
            for i in 0..n {
                for j in 0..m {
                    buf[i * m + j] = objectives[[i, j]] as f32;
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

        // Read back.
        let buffer_slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| format!("map failed: {e:?}"))?;
        let data = buffer_slice.get_mapped_range();
        let f32_results: &[f32] = bytemuck::cast_slice(&data);
        let out: Vec<f64> = f32_results.iter().map(|&v| v as f64).collect();
        drop(data);
        staging_buf.unmap();

        Ok(out)
    }
}
