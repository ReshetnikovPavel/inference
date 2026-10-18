const WEIGHTS_IN_BLOCK: usize = 32;

use super::f16_funcs;

pub struct Q8_0 {
    d: f16_funcs::F16,
    weights: [i8; WEIGHTS_IN_BLOCK],
}

pub fn f32_to_q8_0(v: &[f32]) -> Vec<Q8_0> {
    assert_eq!(v.len() % WEIGHTS_IN_BLOCK, 0);
    let mut res = vec![];
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

        let d = f16_funcs::f16_from_f32(d);
        res.push(Q8_0 { d, weights });
        i += WEIGHTS_IN_BLOCK;
    }
    res
}

fn q8_0_dot(a: &[Q8_0], b: &[Q8_0]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return unsafe { q8_0_dot_avx512f(a, b) };
        }
    }

    q8_0_dot_fallback(a, b)
}

#[target_feature(enable = "avx512f")]
unsafe fn q8_0_dot_avx512f(a: &[Q8_0], b: &[Q8_0]) -> f32 {
    todo!();
}

pub fn q8_0_dot_fallback(a: &[Q8_0], b: &[Q8_0]) -> f32 {
    let mut sum = 0.;
    for i in 0..a.len() {
        let d = f16_funcs::f32_from_f16(a[i].d) * f16_funcs::f32_from_f16(b[i].d);
        let mut block_sum = 0_i32;
        for j in 0..WEIGHTS_IN_BLOCK {
            block_sum += (a[i].weights[j] as i16 * b[i].weights[j] as i16) as i32;
        }
        sum += block_sum as f32 * d;
    }

    sum
}
