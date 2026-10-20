// Standalone benchmark: 6 sign-handling strategies for q8_0-style dot product
// (i8 x i8 -> i32 accumulate), plus bare vpmaddubsw references.
//
// Compile and run with:
//   rustc --edition 2021 -O -C target-cpu=native -o /tmp/q8_dot_variants \
//       benchmarks/q8_0_dot_variants.rs && /tmp/q8_dot_variants
//
// NOT wired into `cargo bench` on purpose: it needs -C target-cpu=native,
// which the crate build doesn't currently apply (rustflags are in Cargo.toml
// [build] instead of .cargo/config.toml).
//
// Results on i5-1135G7 (min of 7 interleaved rounds, 4096 pairs/pass):
//   bare maddubs 256 (ref)              44.6 ns   91.8 GP/s   1.78x
//   bare maddubs 512 (ref)              27.6 ns  148.4 GP/s   2.88x
//   p256 sign+maddubs+madd [baseline]   79.6 ns   51.5 GP/s   1.00x
//   p256 sign + dpbusd (VNNI)  <- WIN  58.9 ns   69.5 GP/s   1.35x
//   p512 abs+masksub sign               78.1 ns   52.4 GP/s   1.02x
//   p512 sign@256 halves                80.4 ns   50.9 GP/s   0.99x
//   p512 widen+dpwssd        [BUGGY]   74.9 ns   54.7 GP/s   1.06x
//   p512 bias+dpbusd (exact)            68.8 ns   59.5 GP/s   1.16x
//
// All pipelines must produce the identical checksum; the widen variant does
// not (see its chunk fn) and its timing is therefore untrustworthy.

use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 4096; // pairs per pass (bytes of each array)
const ITERS: usize = 40000;
const ROUNDS: usize = 7; // interleaved rounds, min per variant

// ---------- helpers ----------
#[inline(always)]
unsafe fn hsum_i16_256(v: __m256i) -> i64 {
    let mut t = [0i16; 16];
    _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
    t.iter().map(|&x| x as i64).sum()
}
#[inline(always)]
unsafe fn hsum_i16_512(v: __m512i) -> i64 {
    let mut t = [0i16; 32];
    _mm512_storeu_si512(t.as_mut_ptr() as *mut __m512i, v);
    t.iter().map(|&x| x as i64).sum()
}
#[inline(always)]
unsafe fn hsum_i32_256(v: __m256i) -> i64 {
    let mut t = [0i32; 8];
    _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
    t.iter().map(|&x| x as i64).sum()
}
#[inline(always)]
unsafe fn hsum_i32_512(v: __m512i) -> i64 {
    let mut t = [0i32; 16];
    _mm512_storeu_si512(t.as_mut_ptr() as *mut __m512i, v);
    t.iter().map(|&x| x as i64).sum()
}

// ---------- bare maddubs references (ceiling/floor) ----------
unsafe fn bare256(a: *const u8, b: *const i8) -> u64 {
    let mut sink = 0u64;
    for _ in 0..ITERS {
        let mut c = [_mm256_setzero_si256(); 4];
        let mut i = 0;
        while i < N {
            for (k, acc) in c.iter_mut().enumerate() {
                let off = i + k * 32;
                let va = _mm256_loadu_si256(a.add(off) as *const __m256i);
                let vb = _mm256_loadu_si256(b.add(off) as *const __m256i);
                *acc = _mm256_add_epi16(*acc, _mm256_maddubs_epi16(va, vb));
            }
            i += 128;
        }
        let mut s = 0i64;
        for acc in c {
            s += hsum_i16_256(acc);
        }
        sink = sink.wrapping_add(s as u64);
    }
    black_box(sink)
}

unsafe fn bare512(a: *const u8, b: *const i8) -> u64 {
    let mut sink = 0u64;
    for _ in 0..ITERS {
        let mut c = [_mm512_setzero_si512(); 4];
        let mut i = 0;
        while i < N {
            for (k, acc) in c.iter_mut().enumerate() {
                let off = i + k * 64;
                let va = _mm512_loadu_si512(a.add(off) as *const __m512i);
                let vb = _mm512_loadu_si512(b.add(off) as *const __m512i);
                *acc = _mm512_add_epi16(*acc, _mm512_maddubs_epi16(va, vb));
            }
            i += 256;
        }
        let mut s = 0i64;
        for acc in c {
            s += hsum_i16_512(acc);
        }
        sink = sink.wrapping_add(s as u64);
    }
    black_box(sink)
}

// ---------- V1: all-256 native sign (baseline) ----------
// 64-pair chunk = two 32-pair bodies -> two i32 accumulators
#[inline(always)]
unsafe fn p256_chunk(a: *const u8, b: *const i8, i: usize, lo: &mut __m256i, hi: &mut __m256i) {
    let ones = _mm256_set1_epi16(1);
    let va = _mm256_loadu_si256(a.add(i) as *const __m256i);
    let vb = _mm256_loadu_si256(b.add(i) as *const __m256i);
    let sa = _mm256_sign_epi8(va, va);
    let sb = _mm256_sign_epi8(vb, va);
    *lo = _mm256_add_epi32(*lo, _mm256_madd_epi16(_mm256_maddubs_epi16(sa, sb), ones));
    let va = _mm256_loadu_si256(a.add(i + 32) as *const __m256i);
    let vb = _mm256_loadu_si256(b.add(i + 32) as *const __m256i);
    let sa = _mm256_sign_epi8(va, va);
    let sb = _mm256_sign_epi8(vb, va);
    *hi = _mm256_add_epi32(*hi, _mm256_madd_epi16(_mm256_maddubs_epi16(sa, sb), ones));
}

unsafe fn p256(a: *const u8, b: *const i8) -> u64 {
    let z = _mm256_setzero_si256();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    let (mut c4, mut c5, mut c6, mut c7) = (z, z, z, z);
    let mut sink = 0u64;
    for _ in 0..ITERS {
        let mut i = 0;
        while i < N {
            p256_chunk(a, b, i, &mut c0, &mut c1);
            p256_chunk(a, b, i + 64, &mut c2, &mut c3);
            p256_chunk(a, b, i + 128, &mut c4, &mut c5);
            p256_chunk(a, b, i + 192, &mut c6, &mut c7);
            i += 256;
        }
        let s = hsum_i32_256(c0)
            + hsum_i32_256(c1)
            + hsum_i32_256(c2)
            + hsum_i32_256(c3)
            + hsum_i32_256(c4)
            + hsum_i32_256(c5)
            + hsum_i32_256(c6)
            + hsum_i32_256(c7);
        sink = sink.wrapping_add(s as u64);
    }
    black_box(sink)
}

// ---------- V2: all-256 native sign + VNNI dpbusd accumulate ----------
#[inline(always)]
unsafe fn p256dpb_chunk(a: *const u8, b: *const i8, i: usize, lo: &mut __m256i, hi: &mut __m256i) {
    let va = _mm256_loadu_si256(a.add(i) as *const __m256i);
    let vb = _mm256_loadu_si256(b.add(i) as *const __m256i);
    let sa = _mm256_sign_epi8(va, va);
    let sb = _mm256_sign_epi8(vb, va);
    *lo = _mm256_dpbusd_epi32(*lo, sa, sb);
    let va = _mm256_loadu_si256(a.add(i + 32) as *const __m256i);
    let vb = _mm256_loadu_si256(b.add(i + 32) as *const __m256i);
    let sa = _mm256_sign_epi8(va, va);
    let sb = _mm256_sign_epi8(vb, va);
    *hi = _mm256_dpbusd_epi32(*hi, sa, sb);
}

unsafe fn p256dpb(a: *const u8, b: *const i8) -> u64 {
    let z = _mm256_setzero_si256();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    let (mut c4, mut c5, mut c6, mut c7) = (z, z, z, z);
    let mut sink = 0u64;
    for _ in 0..ITERS {
        let mut i = 0;
        while i < N {
            p256dpb_chunk(a, b, i, &mut c0, &mut c1);
            p256dpb_chunk(a, b, i + 64, &mut c2, &mut c3);
            p256dpb_chunk(a, b, i + 128, &mut c4, &mut c5);
            p256dpb_chunk(a, b, i + 192, &mut c6, &mut c7);
            i += 256;
        }
        let s = hsum_i32_256(c0)
            + hsum_i32_256(c1)
            + hsum_i32_256(c2)
            + hsum_i32_256(c3)
            + hsum_i32_256(c4)
            + hsum_i32_256(c5)
            + hsum_i32_256(c6)
            + hsum_i32_256(c7);
        sink = sink.wrapping_add(s as u64);
    }
    black_box(sink)
}

// ---------- V3: 512 native sign rebuild (abs + cmpgt + mask-sub) ----------
#[inline(always)]
unsafe fn p512mask_chunk(a: *const u8, b: *const i8, i: usize, acc: &mut __m512i) {
    let ones = _mm512_set1_epi16(1);
    let zero = _mm512_setzero_si512();
    let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
    let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
    let m = _mm512_cmpgt_epi8_mask(zero, va);
    let sa = _mm512_abs_epi8(va);
    let sb = _mm512_mask_sub_epi8(vb, m, zero, vb);
    *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(_mm512_maddubs_epi16(sa, sb), ones));
}

unsafe fn p512mask(a: *const u8, b: *const i8) -> u64 {
    let mut c = [_mm512_setzero_si512(); 4];
    let mut sink = 0u64;
    for _ in 0..ITERS {
        let mut i = 0;
        while i < N {
            p512mask_chunk(a, b, i, &mut c[0]);
            p512mask_chunk(a, b, i + 64, &mut c[1]);
            p512mask_chunk(a, b, i + 128, &mut c[2]);
            p512mask_chunk(a, b, i + 192, &mut c[3]);
            i += 256;
        }
        let mut s = 0i64;
        for acc in c {
            s += hsum_i32_512(acc);
        }
        sink = sink.wrapping_add(s as u64);
    }
    black_box(sink)
}

// ---------- V4: HYBRID — |a| at 512, sign of b at 256 on both halves, maddubs at 512 ----------
#[inline(always)]
unsafe fn p512hyb_chunk(a: *const u8, b: *const i8, i: usize, acc: &mut __m512i) {
    let ones = _mm512_set1_epi16(1);
    let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
    let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
    let sa = _mm512_abs_epi8(va); // |a| wide, one op
    let a_lo = _mm512_castsi512_si256(va);
    let b_lo = _mm512_castsi512_si256(vb);
    let a_hi = _mm512_extracti64x4_epi64::<1>(va);
    let b_hi = _mm512_extracti64x4_epi64::<1>(vb);
    let sb_lo = _mm256_sign_epi8(b_lo, a_lo);
    let sb_hi = _mm256_sign_epi8(b_hi, a_hi);
    let sb = _mm512_inserti64x4::<1>(
        _mm512_inserti64x4::<0>(_mm512_setzero_si512(), sb_lo),
        sb_hi,
    );
    *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(_mm512_maddubs_epi16(sa, sb), ones));
}

unsafe fn p512hyb(a: *const u8, b: *const i8) -> u64 {
    let mut c = [_mm512_setzero_si512(); 4];
    let mut sink = 0u64;
    for _ in 0..ITERS {
        let mut i = 0;
        while i < N {
            p512hyb_chunk(a, b, i, &mut c[0]);
            p512hyb_chunk(a, b, i + 64, &mut c[1]);
            p512hyb_chunk(a, b, i + 128, &mut c[2]);
            p512hyb_chunk(a, b, i + 192, &mut c[3]);
            i += 256;
        }
        let mut s = 0i64;
        for acc in c {
            s += hsum_i32_512(acc);
        }
        sink = sink.wrapping_add(s as u64);
    }
    black_box(sink)
}

// ---------- V5: widen i8->i16 + VNNI dpwssd (no sign trick at all) ----------
// NOTE: [BUGGY] checksum does not match the other variants; known-broken,
// kept only for the timing number. Do not copy this kernel.
#[inline(always)]
unsafe fn p512wide_chunk(a: *const u8, b: *const i8, i: usize, lo: &mut __m512i, hi: &mut __m512i) {
    let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
    let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
    let a_lo = _mm512_castsi512_si256(va);
    let a_hi = _mm512_extracti64x4_epi64::<1>(va);
    let b_lo = _mm512_castsi512_si256(vb);
    let b_hi = _mm512_extracti64x4_epi64::<1>(vb);
    let a16l = _mm512_cvtepi8_epi16(a_lo);
    let a16h = _mm512_cvtepi8_epi16(a_hi);
    let b16l = _mm512_cvtepi8_epi16(b_lo);
    let b16h = _mm512_cvtepi8_epi16(b_hi);
    *lo = _mm512_dpwssd_epi32(*lo, a16l, b16l);
    *hi = _mm512_dpwssd_epi32(*hi, a16h, b16h);
}

unsafe fn p512wide(a: *const u8, b: *const i8) -> u64 {
    let z = _mm512_setzero_si512();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    let (mut c4, mut c5, mut c6, mut c7) = (z, z, z, z);
    let mut sink = 0u64;
    for _ in 0..ITERS {
        let mut i = 0;
        while i < N {
            p512wide_chunk(a, b, i, &mut c0, &mut c1);
            p512wide_chunk(a, b, i + 64, &mut c2, &mut c3);
            p512wide_chunk(a, b, i + 128, &mut c4, &mut c5);
            p512wide_chunk(a, b, i + 192, &mut c6, &mut c7);
            i += 256;
        }
        let s = hsum_i32_512(c0)
            + hsum_i32_512(c1)
            + hsum_i32_512(c2)
            + hsum_i32_512(c3)
            + hsum_i32_512(c4)
            + hsum_i32_512(c5)
            + hsum_i32_512(c6)
            + hsum_i32_512(c7);
        sink = sink.wrapping_add(s as u64);
    }
    black_box(sink)
}

// ---------- V6: bias trick — dpbusd(a^0x80, b) - 128*sum(b) (exact, no sign op) ----------
#[inline(always)]
unsafe fn p512bias_chunk(a: *const u8, b: *const i8, i: usize, acc: &mut __m512i) {
    let ones8 = _mm512_set1_epi8(1);
    let ones16 = _mm512_set1_epi16(1);
    let flip = _mm512_set1_epi8(-128i8); // byte pattern 0x80
    let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
    let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
    let xa = _mm512_xor_si512(va, flip); // a + 128 as u8
    let s = _mm512_dpbusd_epi32(*acc, xa, vb); // sum (a+128)*b
    let pairs = _mm512_maddubs_epi16(ones8, vb); // adjacent sums of b
    let quad = _mm512_madd_epi16(pairs, ones16); // sums of 4 b's
    let corr = _mm512_slli_epi32(quad, 7); // * 128
    *acc = _mm512_sub_epi32(s, corr);
}

unsafe fn p512bias(a: *const u8, b: *const i8) -> u64 {
    let mut c = [_mm512_setzero_si512(); 4];
    let mut sink = 0u64;
    for _ in 0..ITERS {
        let mut i = 0;
        while i < N {
            p512bias_chunk(a, b, i, &mut c[0]);
            p512bias_chunk(a, b, i + 64, &mut c[1]);
            p512bias_chunk(a, b, i + 128, &mut c[2]);
            p512bias_chunk(a, b, i + 192, &mut c[3]);
            i += 256;
        }
        let mut s = 0i64;
        for acc in c {
            s += hsum_i32_512(acc);
        }
        sink = sink.wrapping_add(s as u64);
    }
    black_box(sink)
}

// ---------- driver ----------
type BenchFn = unsafe fn(*const u8, *const i8) -> u64;

fn main() {
    if !is_x86_feature_detected!("avx2")
        || !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512bw")
        || !is_x86_feature_detected!("avx512vl")
        || !is_x86_feature_detected!("avx512vnni")
    {
        eprintln!("CPU missing one of: avx2 avx512f avx512bw avx512vl avx512vnni");
        return;
    }

    let benches: &[(&str, BenchFn, bool)] = &[
        ("bare maddubs 256 (ref)", bare256, false),
        ("bare maddubs 512 (ref)", bare512, false),
        ("p256 sign+maddubs+madd  [baseline]", p256, true),
        ("p256 sign + dpbusd (VNNI)", p256dpb, true),
        ("p512 abs+masksub sign", p512mask, true),
        ("p512 sign@256 halves (yours)", p512hyb, true),
        ("p512 widen+dpwssd  [BUGGY]", p512wide, true),
        ("p512 bias+dpbusd (exact)", p512bias, true),
    ];

    unsafe {
        let layout = std::alloc::Layout::from_size_align(N, 64).unwrap();
        let ap = std::alloc::alloc(layout) as *mut u8;
        let bp = std::alloc::alloc(layout) as *mut u8;
        assert!(!ap.is_null() && !bp.is_null());
        for i in 0..N {
            // a: full i8 range including -128.
            // b: [-64, 62] never -128, so every variant's edge handling agrees
            // (all pipelines must produce the identical checksum).
            *ap.add(i) = (i % 251) as u8;
            *bp.add(i) = (((i * 7) % 127) as i16 - 64) as u8;
        }
        let a = ap as *const u8;
        let b = bp as *const i8;

        // warmup
        for (_, f, _) in benches {
            black_box(f(a, b));
        }

        let mut best = vec![f64::MAX; benches.len()];
        let mut last = vec![0u64; benches.len()];
        for _ in 0..ROUNDS {
            for (k, (_, f, _)) in benches.iter().enumerate() {
                let t0 = Instant::now();
                let r = f(a, b);
                let dt = t0.elapsed().as_secs_f64();
                black_box(r);
                if dt < best[k] {
                    best[k] = dt;
                }
                last[k] = r;
            }
        }

        let base = benches.iter().position(|x| x.0.contains("baseline")).unwrap();
        let base_tps = (N as f64 * ITERS as f64) / best[base];
        println!("{:<36} {:>9} {:>10} {:>8}", "variant", "ns/pass", "GPairs/s", "vs base");
        for (k, (name, _, _)) in benches.iter().enumerate() {
            let tps = (N as f64 * ITERS as f64) / best[k];
            println!(
                "{:<36} {:>9.1} {:>10.1} {:>7.2}x",
                name,
                best[k] * 1e9 / ITERS as f64,
                tps / 1e9,
                tps / base_tps
            );
        }

        // all pipelines must agree exactly
        let pipeline_sums: Vec<u64> = benches
            .iter()
            .enumerate()
            .filter(|(_, x)| x.2)
            .map(|(k, _)| last[k])
            .collect();
        let all_match = pipeline_sums.iter().all(|&x| x == pipeline_sums[0]);
        println!(
            "\npipeline checksums all match: {}  (value {})",
            all_match, pipeline_sums[0]
        );
        if !all_match {
            for (k, (name, _, is_p)) in benches.iter().enumerate() {
                if *is_p {
                    println!("  {} = {}", name, last[k]);
                }
            }
        }

        std::alloc::dealloc(ap, layout);
        std::alloc::dealloc(bp, layout);
    }
}
