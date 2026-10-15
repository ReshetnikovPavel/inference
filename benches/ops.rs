//! Hand-rolled benchmarks for the functions used by `main`.
//!
//! Run with: `cargo bench`
//!
//! No external dependencies are used; each case is timed with
//! `std::time::Instant` and results are fed through `black_box` so the
//! optimizer cannot elide the work being measured.

use std::hint::black_box;
use std::time::Instant;

use inference::{f32_dot, f32_linear, f32_relu, Mat};

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

    // `dot` on the same shape `main` uses.
    let a = [1.; 1000];
    let b = [1.; 1000];
    bench("dot (1000)", || f32_dot(&a, &b));

    // `dot` across a few sizes to show how it scales.
    for n in [100usize, 10_000, 100_000] {
        let a = vec![1.; n];
        let b = vec![1.; n];
        bench(&format!("dot ({n})"), || f32_dot(&a, &b));
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
