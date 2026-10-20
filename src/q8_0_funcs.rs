const WEIGHTS_IN_BLOCK: usize = 32;

use super::f16_funcs;

pub struct Q8_0 {
    d: Vec<f16_funcs::F16>,
    weights: Vec<i8>,
}

impl Q8_0 {
    #[inline]
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

pub fn q8_0_dot(a: &Q8_0, b: &Q8_0) -> f32 {
    debug_assert_eq!(a.weights.len(), b.weights.len());
    debug_assert_eq!(a.d.len(), b.d.len());

    #[cfg(target_arch = "x86_64")]
    {
        // same feature set the kernel declares via #[target_feature]:
        // avx512f for the scales and the conversions, avx512vnni for the
        // `_mm256_dpbusd_epi32` accumulate (which avx512f alone does not
        // guarantee: Skylake-X and Zen 2 have avx512f but no vnni), avx512vl
        // for the 128/256-bit forms.
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vl")
        {
            return unsafe { q8_0_dot_avx512f(a, b) };
        }
    }

    q8_0_dot_fallback(a, b)
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "f16c")]
pub unsafe fn q8_0_dot_avx512f(a: &Q8_0, b: &Q8_0) -> f32 {
    let mut acc = std::arch::x86_64::_mm256_setzero_ps();

    let mut i = 0;
    const BLOCKS_STEP: usize = 16_usize;
    while i + BLOCKS_STEP <= a.d.len() {
        let ad = &a.d[i..i + BLOCKS_STEP];
        let ad = unsafe { std::arch::x86_64::_mm256_loadu_epi16(ad.as_ptr().cast::<i16>()) };
        let ad = std::arch::x86_64::_mm512_cvtph_ps(ad);

        let bd = &b.d[i..i + BLOCKS_STEP];
        let bd = unsafe { std::arch::x86_64::_mm256_loadu_epi16(bd.as_ptr().cast::<i16>()) };
        let bd = std::arch::x86_64::_mm512_cvtph_ps(bd);

        let d = std::arch::x86_64::_mm512_mul_ps(ad, bd);
        let mut ds = [0_f32; BLOCKS_STEP];
        unsafe { std::arch::x86_64::_mm512_storeu_ps(ds.as_mut_ptr(), d) };

        for j in 0..BLOCKS_STEP {
            let a_block = a.weights_block(i + j);
            let b_block = b.weights_block(i + j);

            let a_block = unsafe {
                std::arch::x86_64::_mm256_loadu_si256(
                    a_block.as_ptr() as *const std::arch::x86_64::__m256i
                )
            };
            let b_block = unsafe {
                std::arch::x86_64::_mm256_loadu_si256(
                    b_block.as_ptr() as *const std::arch::x86_64::__m256i
                )
            };
            let a_abs = std::arch::x86_64::_mm256_abs_epi8(a_block);
            let b_times_a_sign = std::arch::x86_64::_mm256_sign_epi8(b_block, a_block);
            let mut dot_product = std::arch::x86_64::_mm256_setzero_si256();
            dot_product = unsafe {
                std::arch::x86_64::_mm256_dpbusd_epi32(dot_product, a_abs, b_times_a_sign)
            };
            let dot_product = std::arch::x86_64::_mm256_cvtepi32_ps(dot_product);
            let d = std::arch::x86_64::_mm256_set1_ps(ds[j]);
            acc = std::arch::x86_64::_mm256_fmadd_ps(d, dot_product, acc);
        }

        i += BLOCKS_STEP;
    }

    while i < a.d.len() {
        let ad = [a.d[i].0; 8];
        let ad = unsafe { std::arch::x86_64::_mm_loadu_epi16(ad.as_ptr().cast::<i16>()) };
        let ad = std::arch::x86_64::_mm256_cvtph_ps(ad);

        let bd = [b.d[i].0; 8];
        let bd = unsafe { std::arch::x86_64::_mm_loadu_epi16(bd.as_ptr().cast::<i16>()) };
        let bd = std::arch::x86_64::_mm256_cvtph_ps(bd);

        let d = std::arch::x86_64::_mm256_mul_ps(ad, bd);

        let a_block = a.weights_block(i);
        let b_block = b.weights_block(i);

        let a_block = unsafe {
            std::arch::x86_64::_mm256_loadu_si256(
                a_block.as_ptr() as *const std::arch::x86_64::__m256i
            )
        };
        let b_block = unsafe {
            std::arch::x86_64::_mm256_loadu_si256(
                b_block.as_ptr() as *const std::arch::x86_64::__m256i
            )
        };
        let a_abs = std::arch::x86_64::_mm256_abs_epi8(a_block);
        let b_times_a_sign = std::arch::x86_64::_mm256_sign_epi8(b_block, a_block);
        let mut dot_product = std::arch::x86_64::_mm256_setzero_si256();
        dot_product = unsafe {
            std::arch::x86_64::_mm256_dpbusd_epi32(dot_product, a_abs, b_times_a_sign)
        };
        let dot_product = std::arch::x86_64::_mm256_cvtepi32_ps(dot_product);
        acc = std::arch::x86_64::_mm256_fmadd_ps(d, dot_product, acc);
        
        i += 1;
    }
    let mut res = [0_f32; 16];
    unsafe { std::arch::x86_64::_mm256_storeu_ps(res.as_mut_ptr(), acc) };
    res.iter().sum()
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
