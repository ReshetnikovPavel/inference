// Is 512-bit f32 execution double-pumped on this CPU?
//
// For i8 integer ops we measured the opposite of the folklore: bare 512-bit
// vpmaddubsw was ~1.4x faster per byte than 256-bit, so "AVX-512 is always
// 2x256 internally" did not hold for the integer side. This benchmark checks
// the f32 side (fmadd + cvtepi32_ps), which uses different execution ports.
//
// 8 independent accumulator chains per iteration to be throughput-bound,
// not latency-bound. Result reported per lane, so 512 vs 256 is apples to
// apples: ratio ~0.5 = double-pumped (no gain from width), ~1.0+ = real width.
//
// rustc --edition 2021 -O -C target-cpu=native -o /tmp/f32_width \
//     benchmarks/f32_width_bench.rs && /tmp/f32_width

use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

const ITERS: usize = 400_000;
const ROUNDS: usize = 7;

#[inline(always)]
unsafe fn hsum256(v: __m256) -> f32 {
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), v);
    t.iter().sum()
}
#[inline(always)]
unsafe fn hsum512(v: __m512) -> f32 {
    let mut t = [0f32; 16];
    _mm512_storeu_ps(t.as_mut_ptr(), v);
    t.iter().sum()
}

// fma at 256: 8 fmadd x 8 lanes = 64 lanes/iter
unsafe fn fma256(a: *const u8) -> f32 {
    let a = a as *const f32;
    let z = _mm256_setzero_ps();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    let (mut c4, mut c5, mut c6, mut c7) = (z, z, z, z);
    for _ in 0..ITERS {
        let v0 = _mm256_loadu_ps(black_box(a));
        let v1 = _mm256_loadu_ps(black_box(a.add(8)));
        c0 = _mm256_fmadd_ps(v0, v0, c0);
        c1 = _mm256_fmadd_ps(v1, v1, c1);
        c2 = _mm256_fmadd_ps(v0, v1, c2);
        c3 = _mm256_fmadd_ps(v1, v0, c3);
        c4 = _mm256_fmadd_ps(v1, v1, c4);
        c5 = _mm256_fmadd_ps(v0, v0, c5);
        c6 = _mm256_fmadd_ps(v1, v0, c6);
        c7 = _mm256_fmadd_ps(v0, v1, c7);
    }
    black_box(hsum256(c0) + hsum256(c1) + hsum256(c2) + hsum256(c3))
        + black_box(hsum256(c4) + hsum256(c5) + hsum256(c6) + hsum256(c7))
}

// fma at 512: 8 fmadd x 16 lanes = 128 lanes/iter
unsafe fn fma512(a: *const u8) -> f32 {
    let a = a as *const f32;
    let z = _mm512_setzero_ps();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    let (mut c4, mut c5, mut c6, mut c7) = (z, z, z, z);
    for _ in 0..ITERS {
        let v0 = _mm512_loadu_ps(black_box(a));
        let v1 = _mm512_loadu_ps(black_box(a.add(16)));
        c0 = _mm512_fmadd_ps(v0, v0, c0);
        c1 = _mm512_fmadd_ps(v1, v1, c1);
        c2 = _mm512_fmadd_ps(v0, v1, c2);
        c3 = _mm512_fmadd_ps(v1, v0, c3);
        c4 = _mm512_fmadd_ps(v1, v1, c4);
        c5 = _mm512_fmadd_ps(v0, v0, c5);
        c6 = _mm512_fmadd_ps(v1, v0, c6);
        c7 = _mm512_fmadd_ps(v0, v1, c7);
    }
    black_box(hsum512(c0) + hsum512(c1) + hsum512(c2) + hsum512(c3))
        + black_box(hsum512(c4) + hsum512(c5) + hsum512(c6) + hsum512(c7))
}

// cvt at 256: 8 cvtepi32_ps x 8 lanes = 64 lanes/iter
unsafe fn cvt256(a: *const u8) -> f32 {
    let a = a as *const i32;
    let z = _mm256_setzero_ps();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    let (mut c4, mut c5, mut c6, mut c7) = (z, z, z, z);
    for _ in 0..ITERS {
        let i0 = _mm256_loadu_si256(black_box(a) as *const __m256i);
        let i1 = _mm256_loadu_si256(black_box(a.add(8)) as *const __m256i);
        let f0 = _mm256_cvtepi32_ps(i0);
        let f1 = _mm256_cvtepi32_ps(i1);
        c0 = _mm256_add_ps(c0, f0);
        c1 = _mm256_add_ps(c1, f1);
        c2 = _mm256_add_ps(c2, f0);
        c3 = _mm256_add_ps(c3, f1);
        c4 = _mm256_add_ps(c4, f0);
        c5 = _mm256_add_ps(c5, f1);
        c6 = _mm256_add_ps(c6, f0);
        c7 = _mm256_add_ps(c7, f1);
    }
    black_box(hsum256(c0) + hsum256(c1) + hsum256(c2) + hsum256(c3))
        + black_box(hsum256(c4) + hsum256(c5) + hsum256(c6) + hsum256(c7))
}

// cvt at 512: 8 cvtepi32_ps x 16 lanes = 128 lanes/iter
unsafe fn cvt512(a: *const u8) -> f32 {
    let a = a as *const i32;
    let z = _mm512_setzero_ps();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    let (mut c4, mut c5, mut c6, mut c7) = (z, z, z, z);
    for _ in 0..ITERS {
        let i0 = _mm512_loadu_si512(black_box(a) as *const __m512i);
        let i1 = _mm512_loadu_si512(black_box(a.add(16)) as *const __m512i);
        let f0 = _mm512_cvtepi32_ps(i0);
        let f1 = _mm512_cvtepi32_ps(i1);
        c0 = _mm512_add_ps(c0, f0);
        c1 = _mm512_add_ps(c1, f1);
        c2 = _mm512_add_ps(c2, f0);
        c3 = _mm512_add_ps(c3, f1);
        c4 = _mm512_add_ps(c4, f0);
        c5 = _mm512_add_ps(c5, f1);
        c6 = _mm512_add_ps(c6, f0);
        c7 = _mm512_add_ps(c7, f1);
    }
    black_box(hsum512(c0) + hsum512(c1) + hsum512(c2) + hsum512(c3))
        + black_box(hsum512(c4) + hsum512(c5) + hsum512(c6) + hsum512(c7))
}

type BenchFn = unsafe fn(*const u8) -> f32;

fn main() {
    if !is_x86_feature_detected!("fma")
        || !is_x86_feature_detected!("avx2")
        || !is_x86_feature_detected!("avx512f")
    {
        eprintln!("CPU missing fma/avx2/avx512f");
        return;
    }

    // lanes processed per iteration of each loop
    let benches: &[(&str, BenchFn, usize)] = &[
        ("fma  256", fma256, 8 * 8),
        ("fma  512", fma512, 8 * 16),
        ("cvt  256", cvt256, 8 * 8),
        ("cvt  512", cvt512, 8 * 16),
    ];

    unsafe {
        let layout = std::alloc::Layout::from_size_align(4096, 64).unwrap();
        let pf = std::alloc::alloc(layout) as *mut f32;
        let pi = std::alloc::alloc(layout) as *mut i32;
        assert!(!pf.is_null() && !pi.is_null());
        for k in 0..1024 {
            // small values: accs stay finite, no constant folding risk
            *pf.add(k) = ((k as f32 % 64.0) - 32.0) * 0.001;
            *pi.add(k) = (k as i32 % 64) - 32;
        }
        let fp = pf as *const u8;
        let ip = pi as *const u8;

        let ptr_for = |name: &str| {
            if name.starts_with("fma") {
                fp
            } else {
                ip
            }
        };

        for (name, f, _) in benches {
            black_box(f(ptr_for(name)));
        }

        let mut best = vec![f64::MAX; benches.len()];
        for _ in 0..ROUNDS {
            for (k, (name, f, _)) in benches.iter().enumerate() {
                let t0 = Instant::now();
                let r = f(ptr_for(name));
                let dt = t0.elapsed().as_secs_f64();
                black_box(r);
                if dt < best[k] {
                    best[k] = dt;
                }
            }
        }

        println!("{:<10} {:>12} {:>12}", "variant", "ns/iter", "GLanes/s");
        let mut rates = Vec::new();
        for (k, (name, _, lanes)) in benches.iter().enumerate() {
            let g = (*lanes as f64 * ITERS as f64) / best[k] / 1e9;
            rates.push(g);
            println!("{:<10} {:>12.2} {:>12.1}", name, best[k] * 1e9 / ITERS as f64, g);
        }
        println!("\nper-lane 512/256 ratio:");
        println!("  fmadd: {:.2}x", rates[1] / rates[0]);
        println!("  cvt:   {:.2}x", rates[3] / rates[2]);

        std::alloc::dealloc(pf as *mut u8, layout);
        std::alloc::dealloc(pi as *mut u8, layout);
    }
}
