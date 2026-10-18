const WEIGHTS_IN_BLOCK: usize = 32;

use super::f16_funcs;

pub struct Q8_0 {
    d: Vec<f16_funcs::F16>,
    weights: Vec<i8>,
}

impl Q8_0 {
    fn weights_block(&self, i: usize) -> &[i8; WEIGHTS_IN_BLOCK] {
        debug_assert!(i < self.d.len());
        unsafe {
            &self.weights[i * WEIGHTS_IN_BLOCK..(i + 1) * WEIGHTS_IN_BLOCK]
                .as_ptr()
                .cast::<[i8; WEIGHTS_IN_BLOCK]>()
                .as_ref_unchecked()
        }
    }
}

pub fn f32_to_q8_0(v: &[f32]) -> Q8_0 {
    assert_eq!(v.len() % WEIGHTS_IN_BLOCK, 0);
    let mut q8_0_d = vec![];
    let mut q8_0_weights = vec![];
    let mut i = 0;
    while i + WEIGHTS_IN_BLOCK <= v.len() {
        let mut weights = [0; WEIGHTS_IN_BLOCK];
        let mut d = 0.;
        for j in 0..WEIGHTS_IN_BLOCK {
            let abs = v[i + j].abs();
            if abs > d {
                d = abs;
            }
        }
        d /= 127.;
        let id = if d == 0. { 0. } else { 1. / d };

        for j in 0..WEIGHTS_IN_BLOCK {
            weights[j] = (v[i + j] * id).round() as i8;
        }
        q8_0_weights.extend_from_slice(&weights);
        q8_0_d.push(f16_funcs::f16_from_f32(d));

        i += WEIGHTS_IN_BLOCK;
    }
    Q8_0 {
        d: q8_0_d,
        weights: q8_0_weights,
    }
}

fn q8_0_dot(a: &Q8_0, b: &Q8_0) -> f32 {
    debug_assert_eq!(a.weights.len(), b.weights.len());
    debug_assert_eq!(a.d.len(), b.d.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return unsafe { q8_0_dot_avx512f(a, b) };
        }
    }

    q8_0_dot_fallback(a, b)
}

#[target_feature(enable = "avx512f")]
#[target_feature(enable = "f16c")]
unsafe fn q8_0_dot_avx512f(a: &Q8_0, b: &Q8_0) -> f32 {
    todo!();
}

pub fn q8_0_dot_fallback(a: &Q8_0, b: &Q8_0) -> f32 {
    let mut sum = 0.;
    for i in 0..a.d.len() {
        let d = f16_funcs::f32_from_f16(a.d[i]) * f16_funcs::f32_from_f16(b.d[i]);
        let mut block_sum = 0_i32;
        let a_block = a.weights_block(i);
        let b_block = b.weights_block(i);
        for j in 0..WEIGHTS_IN_BLOCK {
            block_sum += (a_block[j] as i16 * b_block[j] as i16) as i32;
        }
        sum += block_sum as f32 * d;
    }

    sum
}
