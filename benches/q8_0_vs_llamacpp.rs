//! Our Q8_0 dot products on the exact data llama.cpp's kernel sees.
//!
//! Run with: `cargo bench --bench q8_0_vs_llamacpp`
//!
//! The llama.cpp half lives in `benchmarks/llamacpp_q8_0.c` and prints the
//! same `REF`/`RES` lines for `quantize_row_q8_0_ref` + `ggml_vec_dot_q8_0_q8_0`;
//! run both and merge the outputs to get one table. Two things are compared:
//!
//!   - `quant_hash`: FNV-1a over the quantized blocks. If it matches across
//!     the two programs, our `f32_to_q8_0` produced byte-identical q8_0 to
//!     llama.cpp's quantizer (same amax/127 scale, same rounding, same f16
//!     bits), so the kernel timings below are fed identical inputs.
//!   - `value`: the dot product itself. The two kernels accumulate in a
//!     different order, so equality is expected only up to f32 rounding.
//!
//! Data: xorshift64, seed 0xC0FFEE ^ n, identical to `q8_0_pairs` in ops.rs
//! and to the C program.

use std::hint::black_box;
use std::time::Instant;

use inference::q8_0_funcs::{
    f32_to_q8_0, q8_0_dot_avx512f, q8_0_dot_fallback, q8_0_dot_llamacpp_avx2,
    q8_0_dot_llamacpp_avx2_batched, q8_0_dot_llamacpp_avx2_unrolled, Q8_0,
};

const ITERS: u32 = 10_000;
const WARMUP: u32 = 1_000;
const SIZES: [usize; 9] = [32, 128, 512, 544, 1024, 4096, 16384, 65536, 262144];

/// Same generator as `q8_0_pairs` in ops.rs and as benchmarks/llamacpp_q8_0.c.
/// The shift/normalize steps are exact in f32, so the bits are identical to
/// what the C program computes in float.
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

/// FNV-1a over the blocks in llama.cpp's on-disk order (scale, then its 32
/// weights, per block) — note our in-memory layout is planar (all scales,
/// then all weights), so this also documents the layout difference.
/// Both operands are hashed into one stream, qa first, like the C side.
fn quant_hash(q: &Q8_0, mut h: u64) -> u64 {
    const PRIME: u64 = 1_099_511_628_211;

    let scales = q.scales();
    let weights = q.weights();
    let mut feed = |byte: u8, h: &mut u64| {
        *h ^= byte as u64;
        *h = h.wrapping_mul(PRIME);
    };
    for (i, s) in scales.iter().enumerate() {
        for byte in (s.0 as u16).to_le_bytes() {
            feed(byte, &mut h);
        }
        for &w in &weights[i * 32..i * 32 + 32] {
            feed(w as u8, &mut h);
        }
    }
    h
}

/// Interleaved timing: every implementation is measured once per round on the
/// same data and the best round is reported. Turbo/thermal drift hits all
/// implementations inside the same round, so min-of-rounds is a much fairer
/// A/B on a laptop than one long run each (same trick as
/// benchmarks/q8_0_dot_variants.rs). The C side does the same, so the two
/// programs stay comparable.
const ROUNDS: u32 = 7;

/// `inner` = timed calls per round; the scalar fallback gets few of them
/// because it is orders of magnitude slower.
///
/// Each implementation is measured in its own back-to-back burst inside every
/// round. Note the *ordering* effect: putting the ~40x slower scalar fallback
/// between the vector kernels' bursts evicts them from L1/branch state and
/// inflates their min by ~25%. The fallback is therefore run separately (see
/// `main`), so the kernel-vs-llama.cpp numbers are burst-vs-burst on equal
/// footing.
fn measure(n: usize, mut impls: Vec<(&str, u32, Box<dyn FnMut() -> f32 + '_>)>) {
    for (_, _, f) in impls.iter_mut() {
        for _ in 0..WARMUP {
            black_box(f());
        }
    }

    let mut best = vec![f64::MAX; impls.len()];
    let mut value = vec![0f32; impls.len()];
    for _ in 0..ROUNDS {
        for (k, (_, inner, f)) in impls.iter_mut().enumerate() {
            let start = Instant::now();
            let mut v = 0f32;
            for _ in 0..*inner {
                v = f();
            }
            let dt = start.elapsed().as_secs_f64() * 1e9 / (*inner as f64);
            black_box(v);
            if dt < best[k] {
                best[k] = dt;
            }
            value[k] = v;
        }
    }

    for (k, (name, _, _)) in impls.iter().enumerate() {
        println!(
            "RES n={n} impl={name} value={} ns={:.2} gpairs={:.2}",
            value[k],
            best[k],
            n as f64 / best[k]
        );
    }
}

fn main() {
    let avx512 = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vnni")
        && std::arch::is_x86_feature_detected!("avx512vl");

    println!("# rust q8_0: f32_to_q8_0 + q8_0_dot_fallback / q8_0_dot_avx512f");
    println!("# n impl quant_hash value ns gpairs");

    for &n in &SIZES {
        let mut rng = Rng::new(0xC0FFEE ^ n as u64);
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        rng.fill(&mut a);
        rng.fill(&mut b);

        // f64 reference, same order as the C program
        let ref_dot: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();

        let qa = f32_to_q8_0(&a);
        let qb = f32_to_q8_0(&b);

        let hash = quant_hash(&qb, quant_hash(&qa, 14_695_981_039_346_656_037));
        println!("REF n={n} ref={ref_dot:.17} hash={hash:016x}");

        // The vector kernel and the slow scalar fallback are measured in
        // separate passes: interleaving a 40x slower implementation between
        // bursts inflates the fast one's numbers via cache/branch-state
        // eviction, which would silently be read as "our kernel got slower".
        if avx512 {
            measure(
                n,
                vec![(
                    "rust_avx512",
                    ITERS,
                    Box::new(|| unsafe { q8_0_dot_avx512f(&qa, &qb) }),
                )],
            );
            measure(
                n,
                vec![(
                    "rust_llamacpp_avx2",
                    ITERS,
                    Box::new(|| unsafe { q8_0_dot_llamacpp_avx2(&qa, &qb) }),
                )],
            );
            measure(
                n,
                vec![(
                    "rust_llamacpp_avx2_u4",
                    ITERS,
                    Box::new(|| unsafe { q8_0_dot_llamacpp_avx2_unrolled(&qa, &qb) }),
                )],
            );
            measure(
                n,
                vec![(
                    "rust_llamacpp_avx2_batched",
                    ITERS,
                    Box::new(|| unsafe { q8_0_dot_llamacpp_avx2_batched(&qa, &qb) }),
                )],
            );
        } else {
            println!("# n={n} no avx512f+vnni+vl, kernel skipped");
        }
        measure(
            n,
            vec![(
                "rust_fallback",
                ITERS / 100,
                Box::new(|| q8_0_dot_fallback(&qa, &qb)),
            )],
        );
    }
}
