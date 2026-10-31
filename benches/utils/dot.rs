use std::{fs::File, io::Read as _};

pub fn gen_random_data(len: usize, n: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let total_bytes_needed = len * n * 2;
    let mut raw_bytes = vec![0u8; total_bytes_needed];
    let mut urandom = File::open("/dev/urandom").unwrap();

    urandom.read_exact(&mut raw_bytes).unwrap();

    let mut all_floats = raw_bytes.into_iter().map(|b| (b as i8 as f32) / 128.0);

    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let mut vec_a = Vec::with_capacity(len);
        let mut vec_b = Vec::with_capacity(len);

        for _ in 0..len {
            vec_a.push(all_floats.next().unwrap());
        }
        for _ in 0..len {
            vec_b.push(all_floats.next().unwrap());
        }

        data.push((vec_a, vec_b));
    }

    data
}
