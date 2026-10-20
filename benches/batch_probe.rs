//! How many blocks per iteration should the batched llama-style AVX2 kernel
//! process? `_mm256_cvtph_ps` converts exactly 8 f16 -> 8 f32, so the scale
//! conversion is inherently 1 instruction per 8 blocks. But the *unroll depth*
//! is a separate knob: more groups in flight = more independent chains, at the
//! cost of register pressure and more tail.
//!
//! Run with: `cargo bench --bench batch_probe`

#![allow(unsafe_op_in_unsafe_fn)]

use std::hint::black_box;
use std::time::Instant;

use inference::f16_funcs::F16;
use inference::q8_0_funcs::{f32_to_q8_0, q8_0_dot_avx512f, q8_0_dot_llamacpp_avx2_batched, Q8_0};

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

/// 8 int8 dot-products -> 8 f32 partial sums (llama's exact op mix).
macro_rules! block_q {
    ($wa:expr, $wb:expr, $idx:expr, $ones:expr) => {{
        use std::arch::x86_64::*;
        let qx = _mm256_loadu_si256($wa.add(($idx) * 32) as *const __m256i);
        let qy = _mm256_loadu_si256($wb.add(($idx) * 32) as *const __m256i);
        let ax = _mm256_sign_epi8(qx, qx);
        let sy = _mm256_sign_epi8(qy, qx);
        let dot16 = _mm256_maddubs_epi16(ax, sy);
        let summed = _mm256_madd_epi16($ones, dot16);
        _mm256_cvtepi32_ps(summed)
    }};
}

/// Fold 8 blocks whose combined scales are the 8 lanes of `$ds`.
macro_rules! fold8 {
    ($wa:expr, $wb:expr, $base:expr, $ds:expr, $acc:expr, $ones:expr) => {{
        use std::arch::x86_64::*;
        for j in 0..8usize {
            let q = block_q!($wa, $wb, ($base) + j, $ones);
            let d = _mm256_permutevar8x32_ps($ds, _mm256_set1_epi32(j as i32));
            $acc = _mm256_fmadd_ps(d, q, $acc);
        }
    }};
}

macro_rules! load_ds {
    ($da:expr, $db:expr, $i:expr) => {{
        use std::arch::x86_64::*;
        let sa = _mm_loadu_si128($da.add($i) as *const __m128i);
        let sb = _mm_loadu_si128($db.add($i) as *const __m128i);
        _mm256_mul_ps(_mm256_cvtph_ps(sa), _mm256_cvtph_ps(sb))
    }};
}

/// VNNI version of one block: |x| and sign(y,x), then a single `vpdpbusd`
/// (3 integer ops) instead of llama's vpsignb/vpsignb/vpmaddubsw/vpmaddwd (4).
macro_rules! block_q_vnni {
    ($wa:expr, $wb:expr, $idx:expr) => {{
        use std::arch::x86_64::*;
        let qx = _mm256_loadu_si256($wa.add(($idx) * 32) as *const __m256i);
        let qy = _mm256_loadu_si256($wb.add(($idx) * 32) as *const __m256i);
        let ax = _mm256_abs_epi8(qx);
        let sy = _mm256_sign_epi8(qy, qx);
        let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
        _mm256_cvtepi32_ps(dot)
    }};
}

/// Fold 8 blocks with the VNNI dot.
macro_rules! fold8_vnni {
    ($wa:expr, $wb:expr, $base:expr, $ds:expr, $acc:expr) => {{
        use std::arch::x86_64::*;
        for j in 0..8usize {
            let q = block_q_vnni!($wa, $wb, ($base) + j);
            let d = _mm256_permutevar8x32_ps($ds, _mm256_set1_epi32(j as i32));
            $acc = _mm256_fmadd_ps(d, q, $acc);
        }
    }};
}

macro_rules! kernel {
    ($name:ident, $unroll:expr) => {
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        #[target_feature(enable = "f16c")]
        unsafe fn $name(da: *const F16, wa: *const i8, db: *const F16, wb: *const i8, nb: usize) -> f32 {
            use std::arch::x86_64::*;
            let mut acc = _mm256_setzero_ps();
            let ones = _mm256_set1_epi16(1);
            let mut i = 0;
            while i + $unroll <= nb {
                // convert this group's scales up front, then fold all of them
                let mut ds = [_mm256_setzero_ps(); $unroll / 8];
                for g in 0..($unroll / 8) {
                    ds[g] = load_ds!(da, db, i + g * 8);
                }
                for g in 0..($unroll / 8) {
                    fold8!(wa, wb, i + g * 8, ds[g], acc, ones);
                }
                i += $unroll;
            }
            while i < nb {
                let ds = load_ds!(da, db, i);
                fold8!(wa, wb, i, ds, acc, ones);
                i += 8;
            }
            // nb is a multiple of 8 in this bench; no scalar tail needed.
            let res = _mm256_extractf128_ps(acc, 1);
            let res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
            let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
            let res = _mm_add_ss(res, _mm_movehdup_ps(res));
            _mm_cvtss_f32(res)
        }
    };
}

kernel!(batch8, 8);
kernel!(batch16, 16);
kernel!(batch32, 32);
kernel!(batch64, 64);

macro_rules! kernel_vnni {
    ($name:ident, $unroll:expr) => {
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        #[target_feature(enable = "f16c")]
        #[target_feature(enable = "avx512vnni")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn $name(
            da: *const F16,
            wa: *const i8,
            db: *const F16,
            wb: *const i8,
            nb: usize,
        ) -> f32 {
            use std::arch::x86_64::*;
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + $unroll <= nb {
                let mut ds = [_mm256_setzero_ps(); $unroll / 8];
                for g in 0..($unroll / 8) {
                    ds[g] = load_ds!(da, db, i + g * 8);
                }
                for g in 0..($unroll / 8) {
                    fold8_vnni!(wa, wb, i + g * 8, ds[g], acc);
                }
                i += $unroll;
            }
            while i < nb {
                let ds = load_ds!(da, db, i);
                fold8_vnni!(wa, wb, i, ds, acc);
                i += 8;
            }
            let res = _mm256_extractf128_ps(acc, 1);
            let res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
            let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
            let res = _mm_add_ss(res, _mm_movehdup_ps(res));
            _mm_cvtss_f32(res)
        }
    };
}

kernel_vnni!(batch8_vnni, 8);
kernel_vnni!(batch16_vnni, 16);
kernel_vnni!(batch32_vnni, 32);

/// VNNI dot + batched scales, but with `$nacc` independent accumulators so the
/// `acc = fmadd(d, q, acc)` chain is no longer serial (FMA latency is ~4 cycles,
/// so one accumulator serialises an 8-block group into 32 cycles = 4 cyc/block).
macro_rules! kernel_vnni_macc {
    ($name:ident, $unroll:expr, $nacc:expr) => {
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        #[target_feature(enable = "f16c")]
        #[target_feature(enable = "avx512vnni")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn $name(
            da: *const F16,
            wa: *const i8,
            db: *const F16,
            wb: *const i8,
            nb: usize,
        ) -> f32 {
            use std::arch::x86_64::*;
            let mut accs = [_mm256_setzero_ps(); $nacc];
            let mut i = 0;
            while i + $unroll <= nb {
                let nds = $unroll / 8;
                let mut ds = [_mm256_setzero_ps(); $unroll / 8];
                for g in 0..nds {
                    ds[g] = load_ds!(da, db, i + g * 8);
                }
                for g in 0..nds {
                    for j in 0..8usize {
                        let q = block_q_vnni!(wa, wb, i + g * 8 + j);
                        let d = _mm256_permutevar8x32_ps(ds[g], _mm256_set1_epi32(j as i32));
                        let k = (g * 8 + j) % $nacc;
                        accs[k] = _mm256_fmadd_ps(d, q, accs[k]);
                    }
                }
                i += $unroll;
            }
            while i < nb {
                let ds = load_ds!(da, db, i);
                for j in 0..8usize {
                    let q = block_q_vnni!(wa, wb, i + j);
                    let d = _mm256_permutevar8x32_ps(ds, _mm256_set1_epi32(j as i32));
                    accs[j % $nacc] = _mm256_fmadd_ps(d, q, accs[j % $nacc]);
                }
                i += 8;
            }
            let mut acc = accs[0];
            for k in 1..$nacc {
                acc = _mm256_add_ps(acc, accs[k]);
            }
            let res = _mm256_extractf128_ps(acc, 1);
            let res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
            let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
            let res = _mm_add_ss(res, _mm_movehdup_ps(res));
            _mm_cvtss_f32(res)
        }
    };
}

kernel_vnni_macc!(b8v_a1, 8, 1);
kernel_vnni_macc!(b8v_a2, 8, 2);
kernel_vnni_macc!(b8v_a4, 8, 4);
kernel_vnni_macc!(b16v_a4, 16, 4);
kernel_vnni_macc!(b16v_a8, 16, 8);

/// 16 blocks per iteration, but the scales are converted with AVX512's
/// `_mm512_cvtph_ps` -- 16 f16 -> 16 f32 in ONE instruction, versus two
/// `_mm256_cvtph_ps`. This is the only way to beat the 8-wide conversion
/// quantum of AVX2.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512dq")]
unsafe fn batch16_cvt512(
    da: *const F16,
    wa: *const i8,
    db: *const F16,
    wb: *const i8,
    nb: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
    let mut i = 0;
    while i + 16 <= nb {
        // two conversions for 16 blocks (vs four with _mm256_cvtph_ps)
        let sa = _mm512_cvtph_ps(_mm256_loadu_si256(da.add(i) as *const __m256i));
        let sb = _mm512_cvtph_ps(_mm256_loadu_si256(db.add(i) as *const __m256i));
        let ds = _mm512_mul_ps(sa, sb);
        // split back into the two 8-lane groups, then the usual cheap fold
        let ds_lo = _mm512_castps512_ps256(ds);
        let ds_hi = _mm512_extractf32x8_ps(ds, 1);
        fold8!(wa, wb, i, ds_lo, acc, ones);
        fold8!(wa, wb, i + 8, ds_hi, acc, ones);
        i += 16;
    }
    while i < nb {
        let ds = load_ds!(da, db, i);
        fold8!(wa, wb, i, ds, acc, ones);
        i += 8;
    }
    let res = _mm256_extractf128_ps(acc, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

fn measure<F: FnMut() -> f32>(mut f: F) -> (f64, f32) {
    const WARMUP: u32 = 300;
    const ITERS: u32 = 3_000;
    const ROUNDS: u32 = 9;
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
    let sizes = [1024usize, 4096, 16384, 65536, 262144, 1048576];
    println!("# batched scales; VNNI dot; varying the number of FMA accumulators");
    println!("# n  avx512  b8_avx2  b8_vnni  a1  a2  a4  16a4  16a8   (G-elements/s, best of 9)");
    for &n in &sizes {
        let mut rng = Rng::new(0xC0FFEE ^ n as u64);
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        rng.fill(&mut a);
        rng.fill(&mut b);
        let qa = f32_to_q8_0(&a);
        let qb = f32_to_q8_0(&b);
        let nb = n / 32;
        let (wa, wb) = (qa.weights().as_ptr(), qb.weights().as_ptr());
        let (da, db) = (qa.scales().as_ptr(), qb.scales().as_ptr());

        type K = unsafe fn(*const F16, *const i8, *const F16, *const i8, usize) -> f32;
        let (t512, v512) = measure(|| unsafe { q8_0_dot_avx512f(&qa, &qb) });
        let k8: K = batch8;
        let k8v: K = batch8_vnni;
        let ka1: K = b8v_a1;
        let ka2: K = b8v_a2;
        let ka4: K = b8v_a4;
        let k16a4: K = b16v_a4;
        let k16a8: K = b16v_a8;
        let (t8, v8) = measure(|| unsafe { black_box(k8)(da, wa, db, wb, nb) });
        let (t8v, v8v) = measure(|| unsafe { black_box(k8v)(da, wa, db, wb, nb) });
        let (ta1, va1) = measure(|| unsafe { black_box(ka1)(da, wa, db, wb, nb) });
        let (ta2, va2) = measure(|| unsafe { black_box(ka2)(da, wa, db, wb, nb) });
        let (ta4, va4) = measure(|| unsafe { black_box(ka4)(da, wa, db, wb, nb) });
        let (t16a4, v16a4) = measure(|| unsafe { black_box(k16a4)(da, wa, db, wb, nb) });
        let (t16a8, v16a8) = measure(|| unsafe { black_box(k16a8)(da, wa, db, wb, nb) });

        let gp = |t: f64| n as f64 / t;
        println!(
            "{n:<8} {:.1}  {:.1}  {:.1}  {:.1}  {:.1}  {:.1}  {:.1}  {:.1}",
            gp(t512),
            gp(t8),
            gp(t8v),
            gp(ta1),
            gp(ta2),
            gp(ta4),
            gp(t16a4),
            gp(t16a8),
        );
        for (nm, v) in [
            ("8_avx2", v8),
            ("8_vnni", v8v),
            ("a1", va1),
            ("a2", va2),
            ("a4", va4),
            ("16a4", v16a4),
            ("16a8", v16a8),
            ("512", v512),
        ] {
            assert!((v - v512).abs() / (n as f32) < 1e-2, "n={n} {nm} value {v} vs {v512}");
        }
    }
    align_probe();
}

/// Copy `src` so the returned pointer has address % 64 == `mod64`.
fn buf_at(src: &[i8], mod64: usize) -> (Vec<i8>, *const i8) {
    let mut buf = vec![0i8; src.len() + 128];
    let base = buf.as_ptr() as usize;
    let k = (mod64 + 64 - (base % 64)) % 64;
    buf[k..k + src.len()].copy_from_slice(src);
    let p = buf.as_ptr() as usize + k;
    (buf, p as *const i8)
}

fn align_probe() {
    println!("\n# weight alignment (b8v_a2 kernel), pointer mod 64");
    println!("# n      mod0   mod16   mod8    mod4");
    for &n in &[4096usize, 65536, 262144] {
        let mut rng = Rng::new(0xA11 ^ n as u64);
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        rng.fill(&mut a);
        rng.fill(&mut b);
        let qa = f32_to_q8_0(&a);
        let qb = f32_to_q8_0(&b);
        let nb = n / 32;

        let mut row = Vec::new();
        for &m in &[0usize, 16, 8, 4] {
            let (ba, pa) = buf_at(qa.weights(), m);
            let (bb, pb) = buf_at(qb.weights(), m);
            let (dsa, dsb) = (qa.scales().as_ptr(), qb.scales().as_ptr());
            let k: unsafe fn(*const F16, *const i8, *const F16, *const i8, usize) -> f32 = b8v_a2;
            let (t, _) = measure(|| unsafe { black_box(k)(dsa, pa, dsb, pb, nb) });
            black_box((ba, bb));
            row.push(n as f64 / t);
        }
        println!(
            "{n:<8} {:5.1}  {:5.1}  {:5.1}  {:5.1}",
            row[0], row[1], row[2], row[3]
        );
    }
}
