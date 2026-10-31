use std::arch::is_x86_feature_detected;
use std::arch::x86_64;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { dot_avx512f(a, b) };
        }
    }

    dot_fallback(a, b)
}

pub fn dot_fallback(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

#[target_feature(enable = "avx512f")]
pub unsafe fn dot_avx512f(a: &[f32], b: &[f32]) -> f32 {
    const N: usize = 16;
    let mut acc = x86_64::_mm512_setzero_ps();
    let mut i = 0;
    while i + N <= a.len() {
        unsafe {
            let chunk_a = x86_64::_mm512_loadu_ps(a.as_ptr().add(i));
            let chunk_b = x86_64::_mm512_loadu_ps(b.as_ptr().add(i));
            acc = x86_64::_mm512_fmadd_ps(chunk_a, chunk_b, acc);
        }
        i += N;
    }

    let remaining = a.len() - i;
    if remaining > 0 {
        let mask = (1u16 << remaining) - 1;
        unsafe {
            let chunk_a = x86_64::_mm512_maskz_loadu_ps(mask, a.as_ptr().add(i));
            let chunk_b = x86_64::_mm512_maskz_loadu_ps(mask, b.as_ptr().add(i));
            acc = x86_64::_mm512_fmadd_ps(chunk_a, chunk_b, acc);
        }
    }

    x86_64::_mm512_reduce_add_ps(acc)
}
