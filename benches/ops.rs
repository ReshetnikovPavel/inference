//! Hand-rolled benchmarks for the functions used by `main`.
//!
//! Run with: `cargo bench`
//!
//! No external dependencies are used; each case is timed with
//! `std::time::Instant` and results are fed through `black_box` so the
//! optimizer cannot elide the work being measured.

use std::hint::black_box;
use std::time::{Duration, Instant};

use inference::f32_funcs::{f32_dot_avx512f, f32_dot_fallback, f32_linear, f32_relu, Mat};
use inference::q8_0_funcs::{f32_to_q8_0, q8_0_dot_avx512f, q8_0_dot_fallback, Q8_0};

/// Number of times each case is repeated to average out noise.
const ITERS: u32 = 10_000;
/// Number of untimed warm-up runs before measuring.
const WARMUP: u32 = 1_000;

/// Time `f` without printing; the average per-iteration duration is returned
/// so callers can compare variants of the same operation.
fn time<T>(mut f: impl FnMut() -> T) -> Duration {
    for _ in 0..WARMUP {
        black_box(f());
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        black_box(f());
    }
    start.elapsed() / ITERS
}

/// Time `f`, printing the average per-iteration duration.
fn bench<T>(name: &str, f: impl FnMut() -> T) {
    let per_iter = time(f);
    println!("{name:<28} {per_iter:>12?}/iter");
}

/// Deterministic pseudo-random quantized pairs, so the timed data has a
/// realistic sign/magnitude mix instead of the all-equal `vec![1.; n]`.
fn q8_0_pairs(n: usize, seed: u64) -> (Q8_0, Q8_0) {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let a: Vec<f32> = (0..n)
        .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) * 2. - 1.)
        .collect();
    let b: Vec<f32> = (0..n)
        .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) * 2. - 1.)
        .collect();
    (f32_to_q8_0(&a), f32_to_q8_0(&b))
}

/// Scalar fallback vs the AVX512F kernel, across row lengths.
/// 32 = one block (tail path only), 512 = exactly 16 blocks, 544 = 17 blocks
/// (vector loop + one tail block).
fn bench_q8_0() {
    // same feature set the kernel declares via #[target_feature]
    #[cfg(target_arch = "x86_64")]
    let kernel_ok = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vnni")
        && std::arch::is_x86_feature_detected!("avx512vl");
    #[cfg(not(target_arch = "x86_64"))]
    let kernel_ok = false;

    println!();
    println!(
        "{:<26} {:>12} {:>12} {:>9} {:>12}",
        "q8_0 dot", "naive", "avx512f", "speedup", "GPairs/s"
    );

    for n in [32usize, 128, 512, 544, 1024, 4096, 16384, 65536, 262144] {
        let (qa, qb) = q8_0_pairs(n, 0xC0FFEE ^ n as u64);
        let naive = time(|| q8_0_dot_fallback(&qa, &qb));
        if !kernel_ok {
            println!("{n:<26} {naive:>12?} {:>12} {:>9} {:>12}", "-", "-", "-");
            continue;
        }
        let avx = time(|| unsafe { q8_0_dot_avx512f(&qa, &qb) });
        println!(
            "{n:<26} {naive:>12?} {avx:>12?} {:>8.1}x {:>12.1}",
            naive.as_secs_f64() / avx.as_secs_f64(),
            n as f64 / avx.as_secs_f64() / 1e9
        );
    }
}

fn main() {
    println!("{:<28} {:>12}", "benchmark", "time");

    #[cfg(target_arch = "x86_64")]
    let avx512 = std::arch::is_x86_feature_detected!("avx512f");
    #[cfg(not(target_arch = "x86_64"))]
    let avx512 = false;

    // `dot` on the same shape `main` uses.
    let a = vec![1.; 4096];
    let b = vec![1.; 4096];
    if avx512 {
        bench("f32 avx512 dot (4096)", || unsafe { f32_dot_avx512f(&a, &b) });
    }
    bench("f32 naive dot (4096)", || f32_dot_fallback(&a, &b));

    // `dot` across a few sizes to show how it scales.
    for n in [128usize, 1024, 8192, 65536] {
        let a = vec![1.; n];
        let b = vec![1.; n];
        if avx512 {
            bench(&format!("f32 avx512 dot ({n})"), || unsafe { f32_dot_avx512f(&a, &b) });
        }
        bench(&format!("f32 naive dot ({n})"), || f32_dot_fallback(&a, &b));
    }

    // Q8_0: scalar fallback vs the AVX512F(+VNNI) kernel.
    bench_q8_0();

    // `Mat::new` allocation + construction.
    bench("Mat::new (128x128)", || {
        Mat::new(vec![1.; 128 * 128], 128, 128)
    });

    // Full linear layer: 128 outputs over a 128-wide input.
    let w = Mat::new(vec![1.; 128 * 128], 128, 128);
    let x = vec![1.; 128];
    let bias = vec![1.; 128];
    bench("linear (128x128)", || f32_linear(&w, &x, &bias));

    // `relu` is a single max; time it in bulk so the duration is measurable.
    bench("relu x10_000", || {
        let mut sum = 0.;
        for i in 0..10_000 {
            sum += f32_relu(i as f32 - 5_000.);
        }
        sum
    });
}
