use inference::ops::dot_f32;
use std::hint::black_box;
use std::time::Instant;

#[path = "utils/dot.rs"]
mod utils;

fn main() {
    let lengths = [1024, 1536, 2048, 3072, 3584, 4096, 5120, 8192];
    let runs = 35000;

    let mut avgs = vec![];
    for len in lengths {
        let data = utils::gen_random_data(len, runs);

        let start = Instant::now();
        for (a, b) in data {
            let _ = black_box(dot_f32(black_box(&a), black_box(&b)));
        }
        let duration = start.elapsed();
        let avg = (duration / runs as u32).as_nanos();
        println!("{len:<4} | {avg} ns");
        avgs.push(avg);
    }
    utils::save_results("f32", &lengths, runs, &avgs);
}

