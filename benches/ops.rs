//! Hand-rolled benchmarks for the functions used by `main`.
//!
//! Run with: `cargo bench`
//!
//! No external dependencies are used; each case is timed with
//! `std::time::Instant` and results are fed through `black_box` so the
//! optimizer cannot elide the work being measured.

use std::hint::black_box;
use std::time::Instant;

use inference::f32_funcs::{f32_dot_avx512f, f32_dot_fallback, f32_linear, f32_relu, Mat};
use inference::q8_0_funcs::{f32_to_q8_0, q8_0_dot_fallback};

/// Number of times each case is repeated to average out noise.
const ITERS: u32 = 10_000;
/// Number of untimed warm-up runs before measuring.
const WARMUP: u32 = 1_000;

/// Time `f`, printing the average per-iteration duration.
fn bench<T>(name: &str, mut f: impl FnMut() -> T) {
    for _ in 0..WARMUP {
        black_box(f());
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        black_box(f());
    }
    let per_iter = start.elapsed() / ITERS;

    println!("{name:<28} {per_iter:>12?}/iter");
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
    let qa = f32_to_q8_0(&a);
    let qb = f32_to_q8_0(&b);
    bench("q8_0 naive dot (4096)", || q8_0_dot_fallback(&qa, &qb));

    // `dot` across a few sizes to show how it scales.
    for n in [128usize, 1024, 8192, 65536] {
        let a = vec![1.; n];
        let b = vec![1.; n];
        if avx512 {
            bench(&format!("f32 avx512 dot ({n})"), || unsafe { f32_dot_avx512f(&a, &b) });
        }
        bench(&format!("f32 naive dot ({n})"), || f32_dot_fallback(&a, &b));
        let qa = f32_to_q8_0(&a);
        let qb = f32_to_q8_0(&b);
        bench(&format!("q8_0 naive dot ({n})"), || q8_0_dot_fallback(&qa, &qb));
    }

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
