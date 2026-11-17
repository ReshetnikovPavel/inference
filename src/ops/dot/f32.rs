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

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_i8(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7FFF_FFFF;
                (state % 17) as f32 - 8.0
            })
            .collect()
    }

    #[test]
    fn known_values() {
        assert_eq!(dot_fallback(&[], &[]), 0.0);
        assert_eq!(unsafe { dot_avx512f(&[], &[]) }, 0.0);
        assert_eq!(dot(&[], &[]), 0.0);

        assert_eq!(dot_fallback(&[2.0], &[-3.0]), -6.0);
        assert_eq!(unsafe { dot_avx512f(&[2.0], &[-3.0]) }, -6.0);
        assert_eq!(dot(&[2.0], &[-3.0]), -6.0);

        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot_fallback(&a, &b), 32.0);
        assert_eq!(unsafe { dot_avx512f(&a, &b) }, 32.0);
        assert_eq!(dot(&a, &b), 32.0);
    }

    #[test]
    fn all_lengths_match_fallback() {
        for len in (0..=64).chain([1024]) {
            let a = pseudo_random_i8(len, 1);
            let b = pseudo_random_i8(len, 2);
            let expected = dot_fallback(&a, &b);

            assert_eq!(unsafe { dot_avx512f(&a, &b) }, expected, "len {len}");
            assert_eq!(dot(&a, &b), expected, "len {len}");
        }
    }

    #[test]
    fn unaligned_slices_match_fallback() {
        let a = pseudo_random_i8(64, 3);
        let b = pseudo_random_i8(64, 4);

        for offset in 1..=15 {
            let a = &a[offset..offset + 37];
            let b = &b[offset..offset + 37];
            let expected = dot_fallback(a, b);

            assert_eq!(unsafe { dot_avx512f(a, b) }, expected, "offset {offset}");
            assert_eq!(dot(a, b), expected, "offset {offset}");
        }
    }
}
