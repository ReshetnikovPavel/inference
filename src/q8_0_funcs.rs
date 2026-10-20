const WEIGHTS_IN_BLOCK: usize = 32;

/// Weight bytes are kept cache-line aligned so the 32-byte block loads never
/// straddle a line (a 16-byte-aligned base makes half of them split).
const WEIGHT_ALIGN: usize = 64;

use super::f16_funcs;

pub struct Q8_0 {
    d: Vec<f16_funcs::F16>,
    /// Backing store for the weight bytes; the logical slice starts at
    /// `weights_off` so that it is `WEIGHT_ALIGN`-aligned.
    weights: Vec<i8>,
    weights_off: usize,
}

impl Q8_0 {
    /// Block scales, one per `WEIGHTS_IN_BLOCK` weights.
    pub fn scales(&self) -> &[f16_funcs::F16] {
        &self.d
    }

    /// Quantized weights, `WEIGHTS_IN_BLOCK` per scale, in block order.
    pub fn weights(&self) -> &[i8] {
        let len = self.d.len() * WEIGHTS_IN_BLOCK;
        &self.weights[self.weights_off..self.weights_off + len]
    }

    #[inline]
    fn weights_block(&self, i: usize) -> &[i8; WEIGHTS_IN_BLOCK] {
        debug_assert!(i < self.d.len());
        let weights = self.weights();
        unsafe {
            &weights[i * WEIGHTS_IN_BLOCK..(i + 1) * WEIGHTS_IN_BLOCK]
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
    // Copy into a cache-line-aligned window of the backing store.
    let mut weights = vec![0i8; q8_0_weights.len() + WEIGHT_ALIGN];
    let base = weights.as_ptr() as usize;
    let weights_off = (WEIGHT_ALIGN - base % WEIGHT_ALIGN) % WEIGHT_ALIGN;
    weights[weights_off..weights_off + q8_0_weights.len()].copy_from_slice(&q8_0_weights);

    Q8_0 {
        d: q8_0_d,
        weights,
        weights_off,
    }
}

pub fn q8_0_dot(a: &Q8_0, b: &Q8_0) -> f32 {
    debug_assert_eq!(a.weights().len(), b.weights().len());
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
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vnni")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "f16c")]
pub unsafe fn q8_0_dot_avx512f(a: &Q8_0, b: &Q8_0) -> f32 {
    use std::arch::x86_64::*;

    // One `__m256` of combined scales covers GROUP blocks; the main loop
    // unrolls UNROLL groups (16 blocks) so the f16 loads/conversions batch.
    const GROUP: usize = 8;
    const UNROLL: usize = 2;
    // Independent accumulators. `acc = fmadd(d, q, acc)` is ~4 cycles of
    // latency, so one accumulator serialises each group into a latency-bound
    // chain (~4 cycles/block). NACC chains keep both FMA ports saturated.
    const NACC: usize = 2;

    let nb = a.d.len();
    debug_assert_eq!(nb, b.d.len());
    debug_assert_eq!(a.weights().len(), nb * WEIGHTS_IN_BLOCK);
    debug_assert_eq!(b.weights().len(), nb * WEIGHTS_IN_BLOCK);

    let da = a.d.as_ptr();
    let db = b.d.as_ptr();
    let wa = a.weights().as_ptr();
    let wb = b.weights().as_ptr();

    // SAFETY: `q8_0_dot` checks avx512f + avx512vnni + avx512vl and f16c before
    // calling this, both vectors hold the same number of blocks, and every
    // load below is bounded by `nb` (the scale reads only touch `GROUP`
    // halves when at least `GROUP` blocks remain).
    unsafe {
        let mut accs = [_mm256_setzero_ps(); NACC];
        let mut i = 0usize;

        // 16 blocks per iteration.
        while i + UNROLL * GROUP <= nb {
            let mut ds = [_mm256_setzero_ps(); UNROLL];
            for g in 0..UNROLL {
                let ad = _mm256_cvtph_ps(_mm_loadu_si128(da.add(i + g * GROUP).cast::<__m128i>()));
                let bd = _mm256_cvtph_ps(_mm_loadu_si128(db.add(i + g * GROUP).cast::<__m128i>()));
                ds[g] = _mm256_mul_ps(ad, bd);
            }
            for g in 0..UNROLL {
                for j in 0..GROUP {
                    let off = (i + g * GROUP + j) * WEIGHTS_IN_BLOCK;
                    let qx = _mm256_loadu_si256(wa.add(off).cast::<__m256i>());
                    let qy = _mm256_loadu_si256(wb.add(off).cast::<__m256i>());
                    // |x| * (y * sign(x)) == x * y, and `vpdpbusd` is u8 * i8.
                    let ax = _mm256_abs_epi8(qx);
                    let sy = _mm256_sign_epi8(qy, qx);
                    let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
                    let q = _mm256_cvtepi32_ps(dot);
                    let d = _mm256_permutevar8x32_ps(ds[g], _mm256_set1_epi32(j as i32));
                    let k = (g * GROUP + j) % NACC;
                    accs[k] = _mm256_fmadd_ps(d, q, accs[k]);
                }
            }
            i += UNROLL * GROUP;
        }

        // 8 blocks per iteration.
        while i + GROUP <= nb {
            let ad = _mm256_cvtph_ps(_mm_loadu_si128(da.add(i).cast::<__m128i>()));
            let bd = _mm256_cvtph_ps(_mm_loadu_si128(db.add(i).cast::<__m128i>()));
            let ds = _mm256_mul_ps(ad, bd);
            for j in 0..GROUP {
                let off = (i + j) * WEIGHTS_IN_BLOCK;
                let qx = _mm256_loadu_si256(wa.add(off).cast::<__m256i>());
                let qy = _mm256_loadu_si256(wb.add(off).cast::<__m256i>());
                let ax = _mm256_abs_epi8(qx);
                let sy = _mm256_sign_epi8(qy, qx);
                let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
                let q = _mm256_cvtepi32_ps(dot);
                let d = _mm256_permutevar8x32_ps(ds, _mm256_set1_epi32(j as i32));
                let k = j % NACC;
                accs[k] = _mm256_fmadd_ps(d, q, accs[k]);
            }
            i += GROUP;
        }

        // Scalar tail: fewer than GROUP blocks left, so load each scale on its own
        // instead of reading 8 halves past the end of the scale array.
        while i < nb {
            let ad = f16_bits_to_f32((*da.add(i)).0 as u16);
            let bd = f16_bits_to_f32((*db.add(i)).0 as u16);
            let d = _mm256_set1_ps(ad * bd);

            let off = i * WEIGHTS_IN_BLOCK;
            let qx = _mm256_loadu_si256(wa.add(off).cast::<__m256i>());
            let qy = _mm256_loadu_si256(wb.add(off).cast::<__m256i>());
            let ax = _mm256_abs_epi8(qx);
            let sy = _mm256_sign_epi8(qy, qx);
            let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
            let q = _mm256_cvtepi32_ps(dot);
            accs[0] = _mm256_fmadd_ps(d, q, accs[0]);

            i += 1;
        }

        let mut acc = accs[0];
        for k in 1..NACC {
            acc = _mm256_add_ps(acc, accs[k]);
        }
        let s = _mm256_hadd_ps(acc, acc);
        let s = _mm256_hadd_ps(s, s);
        let lo = _mm256_castps256_ps128(s);
        let hi = _mm256_extractf128_ps(s, 1);
        _mm_cvtss_f32(_mm_add_ss(lo, hi))
    }
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


#[inline]
#[target_feature(enable = "f16c")]
unsafe fn f16_bits_to_f32(h: u16) -> f32 {
    // same as llama.cpp's GGML_CPU_FP16_TO_FP32: _cvtsh_ss
    std::arch::x86_64::_mm_cvtss_f32(std::arch::x86_64::_mm_cvtph_ps(
        std::arch::x86_64::_mm_cvtsi32_si128(h as i32),
    ))
}

/// One block of llama.cpp's `mul_sum_i8_pairs_float`: returns the per-lane
/// combined scale vector and the 8 i32 partial sums as f32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
unsafe fn llamacpp_block(
    da: *const crate::f16_funcs::F16,
    db: *const crate::f16_funcs::F16,
    wa: *const i8,
    wb: *const i8,
    i: usize,
) -> (std::arch::x86_64::__m256, std::arch::x86_64::__m256) {
    use std::arch::x86_64::*;
    let d = _mm256_set1_ps(
        unsafe { f16_bits_to_f32((*da.add(i)).0 as u16) * f16_bits_to_f32((*db.add(i)).0 as u16) },
    );
    let qx = unsafe { _mm256_loadu_si256(wa.add(i * 32) as *const __m256i) };
    let qy = unsafe { _mm256_loadu_si256(wb.add(i * 32) as *const __m256i) };
    let ax = _mm256_sign_epi8(qx, qx);
    let sy = _mm256_sign_epi8(qy, qx);
    let dot16 = _mm256_maddubs_epi16(ax, sy);
    let summed_pairs = _mm256_madd_epi16(_mm256_set1_epi16(1), dot16);
    (d, _mm256_cvtepi32_ps(summed_pairs))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_float_8(acc: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let res = _mm256_extractf128_ps(acc, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

/// A literal transliteration of llama.cpp's AVX2 `ggml_vec_dot_q8_0_q8_0`:
/// one block per iteration, one accumulator.
///
/// Measured, this is ~3x slower than the C original (~7 vs ~21 G-elements/s)
/// *not* because of the planar layout but because LLVM emits the scalar f16
/// scale load as `vpinsrw $0,(scale),%xmm0,%xmmN`, i.e. it picks the
/// accumulator itself as the merge source. That ties every iteration's scale
/// conversion to the previous iteration's FMA, producing a ~18-cycle
/// loop-carried chain. GCC instead uses a dedicated zeroed register. See
/// `benches/layout_probe.rs`; [`q8_0_dot_llamacpp_avx2_unrolled`] is the same
/// algorithm with that dependency hidden.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
pub unsafe fn q8_0_dot_llamacpp_avx2(a: &Q8_0, b: &Q8_0) -> f32 {
    let nb = a.d.len();
    let da = a.d.as_ptr();
    let db = b.d.as_ptr();
    let wa = a.weights().as_ptr();
    let wb = b.weights().as_ptr();

    let mut acc = std::arch::x86_64::_mm256_setzero_ps();
    for i in 0..nb {
        let (d, q) = unsafe { llamacpp_block(da, db, wa, wb, i) };
        acc = std::arch::x86_64::_mm256_fmadd_ps(d, q, acc);
    }
    unsafe { hsum_float_8(acc) }
}

/// The same per-block arithmetic as [`q8_0_dot_llamacpp_avx2`], but four blocks
/// per iteration into four independent accumulators. This hides the false
/// dependency above and reaches llama.cpp's AVX2 throughput (~17-19
/// G-elements/s). Summation order differs only in how the four partial
/// accumulators are combined at the end.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
pub unsafe fn q8_0_dot_llamacpp_avx2_unrolled(a: &Q8_0, b: &Q8_0) -> f32 {
    let nb = a.d.len();
    let da = a.d.as_ptr();
    let db = b.d.as_ptr();
    let wa = a.weights().as_ptr();
    let wb = b.weights().as_ptr();

    let mut acc0 = std::arch::x86_64::_mm256_setzero_ps();
    let mut acc1 = std::arch::x86_64::_mm256_setzero_ps();
    let mut acc2 = std::arch::x86_64::_mm256_setzero_ps();
    let mut acc3 = std::arch::x86_64::_mm256_setzero_ps();

    let mut i = 0;
    while i + 4 <= nb {
        let (d0, q0) = unsafe { llamacpp_block(da, db, wa, wb, i) };
        let (d1, q1) = unsafe { llamacpp_block(da, db, wa, wb, i + 1) };
        let (d2, q2) = unsafe { llamacpp_block(da, db, wa, wb, i + 2) };
        let (d3, q3) = unsafe { llamacpp_block(da, db, wa, wb, i + 3) };
        acc0 = std::arch::x86_64::_mm256_fmadd_ps(d0, q0, acc0);
        acc1 = std::arch::x86_64::_mm256_fmadd_ps(d1, q1, acc1);
        acc2 = std::arch::x86_64::_mm256_fmadd_ps(d2, q2, acc2);
        acc3 = std::arch::x86_64::_mm256_fmadd_ps(d3, q3, acc3);
        i += 4;
    }
    while i < nb {
        let (d, q) = unsafe { llamacpp_block(da, db, wa, wb, i) };
        acc0 = std::arch::x86_64::_mm256_fmadd_ps(d, q, acc0);
        i += 1;
    }

    let acc = std::arch::x86_64::_mm256_add_ps(
        std::arch::x86_64::_mm256_add_ps(acc0, acc1),
        std::arch::x86_64::_mm256_add_ps(acc2, acc3),
    );
    unsafe { hsum_float_8(acc) }
}

/// llama.cpp's exact integer op mix (`vpsignb` / `vpmaddubsw` / `vpmaddwd`),
/// but -- unlike [`q8_0_dot_llamacpp_avx2`] -- the f16 scales are converted
/// eight blocks at a time with a single `vcvtph2ps` on a contiguous load,
/// instead of one scalar conversion per block.
///
/// This is the point of the whole exercise: the naive transliteration is slow
/// because of the *scale conversion*, not the op mix. Vectorising it makes this
/// the fastest kernel measured (fastest than both llama.cpp's own C AVX2 and
/// our AVX512/VNNI kernel), because the per-block scalar `vcvtph2ps` +
/// broadcast is replaced by one instruction per eight blocks and LLVM's
/// accumulator-aliasing `vpinsrw` disappears.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
pub unsafe fn q8_0_dot_llamacpp_avx2_batched(a: &Q8_0, b: &Q8_0) -> f32 {
    use std::arch::x86_64::*;

    let nb = a.d.len();
    let da = a.d.as_ptr();
    let db = b.d.as_ptr();
    let wa = a.weights().as_ptr();
    let wb = b.weights().as_ptr();

    // Two chains: `acc = fmadd(d, q, acc)` is ~4 cycles of latency, so a
    // single accumulator serialises each 8-block group (see the AVX512
    // kernel above for the full explanation).
    let mut accs = [_mm256_setzero_ps(); 2];
    let ones = _mm256_set1_epi16(1);

    let mut i = 0;
    while i + 8 <= nb {
        // 8 contiguous f16 scales -> 8 f32 -> combined d, all vectorised.
        let sa = unsafe { _mm_loadu_si128(da.add(i) as *const __m128i) };
        let sb = unsafe { _mm_loadu_si128(db.add(i) as *const __m128i) };
        let ds = _mm256_mul_ps(_mm256_cvtph_ps(sa), _mm256_cvtph_ps(sb));

        for j in 0..8 {
            let qx = unsafe { _mm256_loadu_si256(wa.add((i + j) * 32) as *const __m256i) };
            let qy = unsafe { _mm256_loadu_si256(wb.add((i + j) * 32) as *const __m256i) };
            let ax = _mm256_sign_epi8(qx, qx);
            let sy = _mm256_sign_epi8(qy, qx);
            let dot16 = _mm256_maddubs_epi16(ax, sy);
            let summed = _mm256_madd_epi16(ones, dot16);
            let q = _mm256_cvtepi32_ps(summed);
            let d = _mm256_permutevar8x32_ps(ds, _mm256_set1_epi32(j as i32));
            let k = j & 1;
            accs[k] = _mm256_fmadd_ps(d, q, accs[k]);
        }
        i += 8;
    }
    while i < nb {
        let (d, q) = unsafe { llamacpp_block(da, db, wa, wb, i) };
        accs[0] = _mm256_fmadd_ps(d, q, accs[0]);
        i += 1;
    }
    unsafe { hsum_float_8(_mm256_add_ps(accs[0], accs[1])) }
}
