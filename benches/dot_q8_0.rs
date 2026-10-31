use inference::ops::dot_q8_0;
use inference::types::q8_0;
use std::hint::black_box;
use std::time::Instant;

#[path = "utils/dot.rs"]
mod utils;

fn main() {
    let lengths = [1024, 1536, 2048, 3072, 3584, 4096, 5120, 8192];
    let runs = 35000;

    for len in lengths {
        let data = utils::gen_random_data(len, runs)
            .into_iter()
            .map(|(a, b)| (q8_0::quantize(&a), q8_0::quantize(&b)))
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (a, b) in data {
            let _ = black_box(dot_q8_0(black_box(&a), black_box(&b)));
        }
        let duration = start.elapsed();
        let avg = duration / runs as u32;
        println!("{len:<4} | {} ns", avg.as_nanos());
    }
}
