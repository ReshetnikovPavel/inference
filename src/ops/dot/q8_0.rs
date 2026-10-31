use std::arch::x86_64;

use crate::types::{
    f16::F16,
    q8_0::{self, VecQ8_0},
};

pub fn dot(a: &VecQ8_0, b: &VecQ8_0) -> f32 {
    debug_assert_eq!(a.weights().len(), b.weights().len());
    debug_assert_eq!(a.d().len(), b.d().len());

    #[cfg(target_arch = "x86_64")]
    {
        // same feature set the kernel declares via #[target_feature]:
        // avx512f for the scales and the conversions, avx512vnni for the
        // `_mm256_dpbusd_epi32` accumulate (which avx512f alone does not
        // guarantee: Skylake-X and Zen 2 have avx512f but no vnni), avx512vl
        // for the 128/256-bit forms.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            return unsafe { dot_simd(a, b) };
        }
    }

    dot_fallback(a, b)
}

pub fn dot_fallback(a: &VecQ8_0, b: &VecQ8_0) -> f32 {
    let mut sum = 0.;
    for i in 0..a.d().len() {
        let d = f32::from(a.d()[i]) * f32::from(b.d()[i]);
        let mut block_sum = 0_i32;
        let a_block = a.weights_block(i);
        let b_block = b.weights_block(i);
        for j in 0..q8_0::WEIGHTS_IN_BLOCK {
            block_sum += (a_block[j] as i16 * b_block[j] as i16) as i32;
        }
        sum += block_sum as f32 * d;
    }

    sum
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "f16c")]
pub fn dot_simd(a: &VecQ8_0, b: &VecQ8_0) -> f32 {
    let blocks_len = a.blocks_len();

    debug_assert_eq!(a.blocks_len(), b.blocks_len());
    debug_assert_eq!(a.weights().len(), blocks_len * q8_0::WEIGHTS_IN_BLOCK);
    debug_assert_eq!(b.weights().len(), blocks_len * q8_0::WEIGHTS_IN_BLOCK);

    // Independent accumulators. `acc = fmadd(d, q, acc)` is ~4 cycles of
    // latency, so one accumulator serialises each group into a latency-bound
    // chain (~4 cycles/block). INDEPENDENT_ACCUMULATORS chains keep both FMA ports saturated.
    const INDEPENDENT_ACCUMULATORS: usize = 2;
    let mut accs = [x86_64::_mm256_setzero_ps(); INDEPENDENT_ACCUMULATORS];

    let da_ptr = a.d().as_ptr();
    let db_ptr = b.d().as_ptr();
    debug_assert!(da_ptr.is_aligned());
    debug_assert!(db_ptr.is_aligned());

    let mut i = 0_usize;

    const UNROLL: usize = 2;
    const F32_FIT_IN_MM256: usize = 256 / 32;

    while i + UNROLL * F32_FIT_IN_MM256 <= blocks_len {
        let mut ds = [x86_64::_mm256_setzero_ps(); UNROLL];
        for j in 0..UNROLL {
            ds[j] = unsafe { mm256_mul_d(a.d(), b.d(), i + j * F32_FIT_IN_MM256) };
        }
        for j in 0..UNROLL {
            for k in 0..F32_FIT_IN_MM256 {
                let d_index = x86_64::_mm256_set1_epi32(k as i32);
                let d = x86_64::_mm256_permutevar8x32_ps(ds[j], d_index);

                let dot = unsafe {
                    mm256_mul_weights(a.weights(), b.weights(), i + j * F32_FIT_IN_MM256 + k)
                };

                let accs_index = (j * F32_FIT_IN_MM256 + k) % INDEPENDENT_ACCUMULATORS;
                accs[accs_index] = x86_64::_mm256_fmadd_ps(d, dot, accs[accs_index]);
            }
        }
        i += UNROLL * F32_FIT_IN_MM256;
    }

    while i + F32_FIT_IN_MM256 <= blocks_len {
        let ds = unsafe { mm256_mul_d(a.d(), b.d(), i) };
        for j in 0..F32_FIT_IN_MM256 {
            let d_index = x86_64::_mm256_set1_epi32(j as i32);
            let d = x86_64::_mm256_permutevar8x32_ps(ds, d_index);

            let dot = unsafe { mm256_mul_weights(a.weights(), b.weights(), i + j) };

            let accs_index = j % INDEPENDENT_ACCUMULATORS;
            accs[accs_index] = x86_64::_mm256_fmadd_ps(d, dot, accs[accs_index]);
        }
        i += F32_FIT_IN_MM256;
    }

    while i < blocks_len {
        let da = f32::from(unsafe { (*da_ptr.add(i)).0 });
        let db = f32::from(unsafe { (*db_ptr.add(i)).0 });
        let d = x86_64::_mm256_set1_ps(da * db);

        let dot = unsafe { mm256_mul_weights(a.weights(), b.weights(), i) };

        accs[0] = x86_64::_mm256_fmadd_ps(d, dot, accs[0]);

        i += 1;
    }

    let mut acc = accs[0];
    for k in 1..INDEPENDENT_ACCUMULATORS {
        acc = x86_64::_mm256_add_ps(acc, accs[k]);
    }
    let sum = mm256_hadd_all(acc);
    x86_64::_mm_cvtss_f32(sum)
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "f16c")]
#[inline]
fn mm256_hadd_all(a: x86_64::__m256) -> x86_64::__m128 {
    //  A     B     C     D    |  E     F     G     H
    let s = x86_64::_mm256_hadd_ps(a, a);
    //  A+B   C+D   A+B   C+D  |  E+F   G+H   E+F   G+H
    let s = x86_64::_mm256_hadd_ps(s, s);
    //  ABCD  ABCD  ABCD  ABCD |  EFGH  EFGH  EFGH  EFGH
    let lo = x86_64::_mm256_castps256_ps128(s);
    // ABCD
    let hi = x86_64::_mm256_extractf128_ps(s, 1);
    // EFGH
    x86_64::_mm_add_ss(lo, hi)
    //  ABCD  +  EFGH
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "f16c")]
#[inline]
unsafe fn mm256_mul_d(da: &[F16], db: &[F16], i_d_block: usize) -> x86_64::__m256 {
    let da_ptr = da.as_ptr();
    let db_ptr = db.as_ptr();
    debug_assert!(da_ptr.is_aligned());
    debug_assert!(db_ptr.is_aligned());
    let da = unsafe { x86_64::_mm_load_si128(da_ptr.add(i_d_block).cast::<x86_64::__m128i>()) };
    let db = unsafe { x86_64::_mm_load_si128(db_ptr.add(i_d_block).cast::<x86_64::__m128i>()) };
    let da = x86_64::_mm256_cvtph_ps(da);
    let db = x86_64::_mm256_cvtph_ps(db);
    x86_64::_mm256_mul_ps(da, db)
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "f16c")]
#[inline]
unsafe fn mm256_mul_weights(wa: &[i8], wb: &[i8], q8_0_block_i: usize) -> x86_64::__m256 {
    let offset = q8_0_block_i * q8_0::WEIGHTS_IN_BLOCK;
    let a_block_ptr = unsafe { wa.as_ptr().add(offset).cast::<x86_64::__m256i>() };
    let b_block_ptr = unsafe { wb.as_ptr().add(offset).cast::<x86_64::__m256i>() };
    debug_assert!(a_block_ptr.is_aligned());
    debug_assert!(b_block_ptr.is_aligned());
    let a_block = unsafe { x86_64::_mm256_load_si256(a_block_ptr) };
    let b_block = unsafe { x86_64::_mm256_load_si256(b_block_ptr) };
    let dot = unsafe { mm256_i8_mul_i8(a_block, b_block) };
    x86_64::_mm256_cvtepi32_ps(dot)
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "f16c")]
#[inline]
unsafe fn mm256_i8_mul_i8(x: x86_64::__m256i, y: x86_64::__m256i) -> x86_64::__m256i {
    // |x| * (y * sign(x)) == x * y, and `vpdpbusd` is u8 * i8.
    let x_abs = x86_64::_mm256_abs_epi8(x);
    let y_mul_x_sign = x86_64::_mm256_sign_epi8(y, x);
    let zeros = x86_64::_mm256_setzero_si256();
    x86_64::_mm256_dpbusd_epi32(zeros, x_abs, y_mul_x_sign)
}
