//! Experiment: is the Rust transliteration of llama.cpp's AVX2 q8_0 kernel
//! slower because of (a) the planar vs interleaved block layout, or (b) LLVM's
//! instruction selection introducing a loop-carried false dependency? Answer:
//! (b). `planar` ~= `inter`, while `inter_u4` / `inter_batch` (same math,
//! dependency hidden) reach llama.cpp's C throughput.
//!
//! Run with: `cargo bench --bench layout_probe`

use std::hint::black_box;
use std::time::Instant;

use inference::q8_0_funcs::{f32_to_q8_0, q8_0_dot_llamacpp_avx2, Q8_0};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn fill(&mut self, dst: &mut [f32]) {
        for v in dst {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            *v = (x >> 40) as f32 / (1u64 << 24) as f32 * 2. - 1.;
        }
    }
}

/// Interleaved copy: 2 bytes of f16 scale + 32 weights per block (llama.cpp's
/// block_q8_0 layout, stride 34).
fn interleave(q: &Q8_0) -> Vec<u8> {
    let scales = q.scales();
    let weights = q.weights();
    let mut out = Vec::with_capacity(scales.len() * 34);
    for (i, s) in scales.iter().enumerate() {
        out.extend_from_slice(&(s.0 as u16).to_le_bytes());
        for &w in &weights[i * 32..i * 32 + 32] {
            out.push(w as u8);
        }
    }
    out
}

#[inline]
#[target_feature(enable = "f16c")]
unsafe fn f16_bits_to_f32(h: u16) -> f32 {
    std::arch::x86_64::_mm_cvtss_f32(std::arch::x86_64::_mm_cvtph_ps(
        std::arch::x86_64::_mm_cvtsi32_si128(h as i32),
    ))
}

macro_rules! block_dot {
    ($pa:expr, $pb:expr, $off:expr, $ones:expr) => {{
        use std::arch::x86_64::*;
        let dx = f16_bits_to_f32(*($pa.add($off) as *const u16));
        let dy = f16_bits_to_f32(*($pb.add($off) as *const u16));
        let d = _mm256_set1_ps(dx * dy);
        let qx = _mm256_loadu_si256($pa.add($off + 2) as *const __m256i);
        let qy = _mm256_loadu_si256($pb.add($off + 2) as *const __m256i);
        let ax = _mm256_sign_epi8(qx, qx);
        let sy = _mm256_sign_epi8(qy, qx);
        let dot16 = _mm256_maddubs_epi16(ax, sy);
        let summed = _mm256_madd_epi16($ones, dot16);
        let q = _mm256_cvtepi32_ps(summed);
        (d, q)
    }};
}
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
unsafe fn dot_interleaved(pa: *const u8, pb: *const u8, nb: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
    for i in 0..nb {
        let (d, q) = block_dot!(pa, pb, i * 34, ones);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    hsum(acc)
}

/// Single accumulator (same accumulation order as llama.cpp), but scales for a
/// group of 8 blocks are converted with one `_mm256_cvtph_ps` instead of a
/// per-block scalar conversion, so LLVM cannot create the false dependency.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
unsafe fn dot_interleaved_batch(pa: *const u8, pb: *const u8, nb: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
    let mut i = 0;
    while i + 8 <= nb {
        // gather the 8 f16 scales from the 34-byte stride into a 128-bit reg
        let mut sa = [0u16; 8];
        let mut sb = [0u16; 8];
        for j in 0..8 {
            sa[j] = *(pa.add((i + j) * 34) as *const u16);
            sb[j] = *(pb.add((i + j) * 34) as *const u16);
        }
        let sa = _mm_loadu_si128(sa.as_ptr() as *const __m128i);
        let sb = _mm_loadu_si128(sb.as_ptr() as *const __m128i);
        let da = _mm256_cvtph_ps(sa);
        let db = _mm256_cvtph_ps(sb);
        let ds = _mm256_mul_ps(da, db);
        for j in 0..8 {
            let (_, q) = block_dot!(pa, pb, (i + j) * 34, ones);
            let d = _mm256_permutevar8x32_ps(ds, _mm256_set1_epi32(j as i32));
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        i += 8;
    }
    while i < nb {
        let (d, q) = block_dot!(pa, pb, i * 34, ones);
        acc = _mm256_fmadd_ps(d, q, acc);
        i += 1;
    }
    hsum(acc)
}

/// llama.cpp's exact integer op mix (vpsignb/vpmaddubsw/vpmaddwd) on *our
/// planar* layout, but scales are converted 8-at-a-time with contiguous
/// `_mm_loadu_si128` -> `_mm256_cvtph_ps` (no per-block scalar conversion, no
/// gather). This isolates "the op mix" from "the scalar scale conversion".
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
unsafe fn dot_planar_batch8(
    da: *const inference::f16_funcs::F16,
    wa: *const i8,
    db: *const inference::f16_funcs::F16,
    wb: *const i8,
    nb: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
    let mut i = 0;
    while i + 8 <= nb {
        let sa = _mm_loadu_si128(da.add(i) as *const __m128i);
        let sb = _mm_loadu_si128(db.add(i) as *const __m128i);
        let ds = _mm256_mul_ps(_mm256_cvtph_ps(sa), _mm256_cvtph_ps(sb));
        for j in 0..8 {
            let qx = _mm256_loadu_si256(wa.add((i + j) * 32) as *const __m256i);
            let qy = _mm256_loadu_si256(wb.add((i + j) * 32) as *const __m256i);
            let ax = _mm256_sign_epi8(qx, qx);
            let sy = _mm256_sign_epi8(qy, qx);
            let dot16 = _mm256_maddubs_epi16(ax, sy);
            let summed = _mm256_madd_epi16(ones, dot16);
            let q = _mm256_cvtepi32_ps(summed);
            let d = _mm256_permutevar8x32_ps(ds, _mm256_set1_epi32(j as i32));
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        i += 8;
    }
    while i < nb {
        let dx = f16_bits_to_f32((*da.add(i)).0 as u16);
        let dy = f16_bits_to_f32((*db.add(i)).0 as u16);
        let d = _mm256_set1_ps(dx * dy);
        let qx = _mm256_loadu_si256(wa.add(i * 32) as *const __m256i);
        let qy = _mm256_loadu_si256(wb.add(i * 32) as *const __m256i);
        let ax = _mm256_sign_epi8(qx, qx);
        let sy = _mm256_sign_epi8(qy, qx);
        let dot16 = _mm256_maddubs_epi16(ax, sy);
        let summed = _mm256_madd_epi16(ones, dot16);
        let q = _mm256_cvtepi32_ps(summed);
        acc = _mm256_fmadd_ps(d, q, acc);
        i += 1;
    }
    hsum(acc)
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
unsafe fn dot_interleaved_u4(pa: *const u8, pb: *const u8, nb: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
    let mut i = 0;
    while i + 4 <= nb {
        let (d0, q0) = block_dot!(pa, pb, i * 34, ones);
        let (d1, q1) = block_dot!(pa, pb, (i + 1) * 34, ones);
        let (d2, q2) = block_dot!(pa, pb, (i + 2) * 34, ones);
        let (d3, q3) = block_dot!(pa, pb, (i + 3) * 34, ones);
        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);
        acc2 = _mm256_fmadd_ps(d2, q2, acc2);
        acc3 = _mm256_fmadd_ps(d3, q3, acc3);
        i += 4;
    }
    while i < nb {
        let (d, q) = block_dot!(pa, pb, i * 34, ones);
        acc0 = _mm256_fmadd_ps(d, q, acc0);
        i += 1;
    }
    hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum(acc: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let res = _mm256_extractf128_ps(acc, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

fn measure<F: FnMut() -> f32>(mut f: F) -> (f64, f32) {
    const WARMUP: u32 = 1000;
    const ITERS: u32 = 10_000;
    const ROUNDS: u32 = 7;
    for _ in 0..WARMUP {
        black_box(f());
    }
    let mut best = f64::MAX;
    let mut v = 0f32;
    for _ in 0..ROUNDS {
        let t = Instant::now();
        for _ in 0..ITERS {
            v = f();
        }
        let dt = t.elapsed().as_secs_f64() * 1e9 / ITERS as f64;
        if dt < best {
            best = dt;
        }
    }
    (best, v)
}

fn main() {
    let sizes = [1024usize, 4096, 16384, 65536, 262144];
    println!("# planar vs interleaved vs interleaved-u4, same AVX2 algorithm");
    for &n in &sizes {
        let mut rng = Rng::new(0xC0FFEE ^ n as u64);
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        rng.fill(&mut a);
        rng.fill(&mut b);
        let qa = f32_to_q8_0(&a);
        let qb = f32_to_q8_0(&b);
        let nb = n / 32;
        let ia = interleave(&qa);
        let ib = interleave(&qb);

        let (planar, vp) = measure(|| unsafe { q8_0_dot_llamacpp_avx2(&qa, &qb) });
        let f: unsafe fn(*const u8, *const u8, usize) -> f32 = dot_interleaved;
        let (inter, vi) = measure(|| unsafe { black_box(f)(ia.as_ptr(), ib.as_ptr(), nb) });
        let f4: unsafe fn(*const u8, *const u8, usize) -> f32 = dot_interleaved_u4;
        let (u4, v4) = measure(|| unsafe { black_box(f4)(ia.as_ptr(), ib.as_ptr(), nb) });
        let fb: unsafe fn(*const u8, *const u8, usize) -> f32 = dot_interleaved_batch;
        let (batch, vb) = measure(|| unsafe { black_box(fb)(ia.as_ptr(), ib.as_ptr(), nb) });

        let wa = qa.weights().as_ptr();
        let wb = qb.weights().as_ptr();
        let da = qa.scales().as_ptr();
        let db = qb.scales().as_ptr();
        let fp: unsafe fn(
            *const inference::f16_funcs::F16,
            *const i8,
            *const inference::f16_funcs::F16,
            *const i8,
            usize,
        ) -> f32 = dot_planar_batch8;
        let (planar_b8, vpb) =
            measure(|| unsafe { black_box(fp)(da, wa, db, wb, nb) });

        println!(
            "n={n:<7} planar={planar:>7.1}ns({:.1}gp) inter={inter:>7.1}ns({:.1}gp) inter_u4={u4:>7.1}ns({:.1}gp) inter_batch={batch:>7.1}ns({:.1}gp) planar_b8={planar_b8:>7.1}ns({:.1}gp)",
            n as f64 / planar,
            n as f64 / inter,
            n as f64 / u4,
            n as f64 / batch,
            n as f64 / planar_b8,
        );
        assert!(
            (vp - vi).abs() < 1e-3
                && (vp - v4).abs() < 1e-2
                && (vp - vb).abs() < 1e-2
                && (vp - vpb).abs() < 1e-2,
            "value mismatch"
        );
    }
}
