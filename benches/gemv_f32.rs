use std::hint::black_box;
use std::time::Instant;

use inference::ops::gemv_f32;

#[path = "utils/gemv.rs"]
mod utils;

fn main() {
    let dimensions = [
        (1536, 1536),
        (8960, 1536),
        (3584, 3584),
        // (14336, 4096),
        // (4096, 14336),
        // (8192, 8192),
        // (29568, 8192),
        // (151936, 2048),
    ];
    let runs = 1000;

    let mut avgs = vec![];
    for (m, n) in dimensions {
        let (w, vectors) = utils::gen_random_data(m, n, runs);

        let start = Instant::now();
        for (x, b) in vectors {
            let _ = black_box(gemv_f32(black_box(&w), black_box(&x), black_box(&b)));
        }
        let duration = start.elapsed();
        let avg = (duration / runs as u32).as_nanos();
        println!("{m:<6} x {n:<6} | {avg} ns");
        avgs.push(avg);
    }
    utils::save_results_gemv("f32", &dimensions, runs, &avgs);
}
