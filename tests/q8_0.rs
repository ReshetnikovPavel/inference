// Integration tests for Q8_0 quantization.
//
// These use only the public API: f32_to_q8_0 + q8_0_dot_fallback,
// compared against the f32 reference dot products.
//
// NOTE: tests compare with a *relative tolerance*, never exact equality,
// because quantization is lossy by design. Exact equality is only used in
// the one case where the values are chosen to be perfectly representable.

use inference::q8_0_funcs::{f32_to_q8_0, q8_0_dot_fallback, Q8_0};
use inference::f32_funcs::{f32_dot, f32_dot_fallback};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// xorshift64 PRNG — deterministic so failures are reproducible without
/// pulling in a `rand` dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1) // xorshift must not start at 0
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as f32; // top 24 bits
        mantissa / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

/// Sum of |a_i * b_i| — a cancellation-free yardstick for the dot product.
/// (Comparing against the dot itself is useless when random vectors nearly
/// cancel to zero: the relative error would explode for no real reason.)
fn l1_products(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 * y as f64).abs())
        .sum()
}

/// Relative error of `got` vs `want`, normalized by the scale of the inputs.
fn rel_err(got: f32, want: f32, yardstick: f64) -> f64 {
    (got as f64 - want as f64).abs() / yardstick
}

fn quantized_dot(a: &[f32], b: &[f32]) -> f32 {
    let qa = f32_to_q8_0(a);
    let qb = f32_to_q8_0(b);
    q8_0_dot_fallback(&qa, &qb)
}

// ---------------------------------------------------------------------------
// 1. exact case: everything is a multiple of 2^-10, so d = 2^-10 is exactly
//    representable in f16 AND in f32, the int8 values come out as exact
//    integers, and the whole dot product must be bit-exact vs f32.
// ---------------------------------------------------------------------------

#[test]
fn exact_when_scale_is_power_of_two() {
    // values = integer * 2^-10, with one element at exactly ±127 * 2^-10
    // so that amax / 127 == 2^-10 exactly.
    let a: Vec<f32> = (0..32i32).map(|i| (i - 63) as f32 / 1024.0).collect();
    let b: Vec<f32> = (0..32i32).map(|i| (63 - i) as f32 / 1024.0).collect();
    // a[0] = -63/1024 ... bump one element to make amax == 127*2^-10
    let mut a = a;
    let mut b = b;
    a[0] = 127.0 / 1024.0;
    b[0] = -127.0 / 1024.0;

    let want = f32_dot_fallback(&a, &b);
    let got = quantized_dot(&a, &b);

    assert_eq!(got, want, "quantized dot must be exact for dyadic values");
}

// ---------------------------------------------------------------------------
// 2. regression: the /128-vs-/127 saturation bug and the divide-vs-multiply
//    scale bug both blew this simple case wide open.
// ---------------------------------------------------------------------------

#[test]
fn all_ones_matches_reference_within_f16_noise() {
    let a = vec![1.0f32; 4096];
    let b = vec![1.0f32; 4096];

    let want = f32_dot(&a, &b);
    let got = quantized_dot(&a, &b);

    assert_eq!(want, 4096.0);
    let err = rel_err(got, want, 4096.0);
    // f16 storage of d costs up to ~2^-11 per scale, two scales -> ~1e-3.
    assert!(err < 1e-3, "rel err = {err} (got {got}, want {want})");
}

// ---------------------------------------------------------------------------
// 3. random data across sizes and magnitudes — the main accuracy property.
// ---------------------------------------------------------------------------

#[test]
fn random_vectors_stay_within_1pct() {
    // Magnitudes are kept >= 1e-4 on purpose: Q8_0 stores d as f16, whose
    // smallest subnormal is ~6e-8. A block with amax/127 below that flushes
    // the scale to zero (see the dedicated test at the bottom).
    for &scale in &[1e-2f32, 1.0, 1e2, 1e5] {
        for &n in &[32usize, 4096, 4096 * 3 + 64] {
            let mut rng = Rng::new(0xC0FFEE ^ (scale.to_bits() as u64) ^ n as u64);
            let a = rng.vec(n, scale);
            let b = rng.vec(n, scale);

            let want = f32_dot(&a, &b);
            let want_scalar = f32_dot_fallback(&a, &b);
            let got = quantized_dot(&a, &b);

            let yardstick = l1_products(&a, &b);

            // sanity: the two f32 references agree (up to summation-order
            // rounding — avx512 accumulates across lanes, scalar does not,
            // and float addition is not associative, so never assert_eq here)
            assert!(
                rel_err(want_scalar, want, yardstick.max(f64::MIN_POSITIVE)) < 1e-6,
                "avx512 vs scalar f32 disagree: {want} vs {want_scalar}"
            );

            let err = rel_err(got, want, yardstick);
            assert!(
                err < 1e-2,
                "scale={scale} n={n}: rel err = {err} (got {got}, want {want})",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. zero blocks: the all-zero guard must produce no NaN and contribute 0.
// ---------------------------------------------------------------------------

#[test]
fn zero_block_is_exactly_zero_and_not_nan() {
    let mut rng = Rng::new(42);
    let half_a = rng.vec(32, 1.0);
    let half_b = rng.vec(32, 1.0);

    // [ zero block | real block ]
    let mut a = vec![0.0f32; 32];
    a.extend_from_slice(&half_a);
    let mut b = vec![0.0f32; 32];
    b.extend_from_slice(&half_b);

    // quantize the two halves separately: the zero block must contribute
    // exactly nothing, so full [zero|half] must equal plain [half].
    let got = quantized_dot(&a, &b);
    let want_q = quantized_dot(&half_a, &half_b);

    assert!(!got.is_nan(), "zero block leaked NaN into the result");
    // exact: an all-zero block's int8 weights are all 0, so it adds 0.0.
    assert_eq!(got, want_q);

    // and it must stay close to the unquantized reference, too
    let want = f32_dot_fallback(&half_a, &half_b);
    let err = rel_err(got, want, l1_products(&half_a, &half_b));
    assert!(err < 1e-2, "rel err vs f32 = {err}");
}

// ---------------------------------------------------------------------------
// 5. outliers: one huge value next to tiny ones (max maps to ±127).
// ---------------------------------------------------------------------------

#[test]
fn outlier_block_dominates_and_does_not_saturate_silently() {
    let mut a = vec![1e-3f32; 32];
    a[0] = 1e3;
    let mut b = vec![1e-3f32; 32];
    b[0] = -1e3;

    let want = f32_dot(&a, &b);
    let got = quantized_dot(&a, &b);

    let yardstick = l1_products(&a, &b);
    let err = rel_err(got, want, yardstick);
    assert!(err < 1e-2, "rel err = {err} (got {got}, want {want})");

    // the tiny weights are ~1e6 times smaller than the outlier, so they
    // round to 0 under this block's scale — that's expected, not a bug.
    // the dot must still be negative (driven by the outlier).
    assert!(got < 0.0, "outlier sign lost: {got}");
}

// ---------------------------------------------------------------------------
// 6. the format contract: row length must divide by 32 (llama.cpp asserts
//    the same thing, no padding).
// ---------------------------------------------------------------------------

#[test]
#[should_panic]
fn rejects_length_not_divisible_by_block() {
    f32_to_q8_0(&[0.5f32; 33]);
}

#[test]
fn accepts_length_equal_to_block() {
    // Exactly one block: the quantized dot must round-trip the dyadic values
    // (32 * 0.5 * 0.5 == 8.0) instead of panicking on the length check.
    let got = quantized_dot(&[0.5f32; 32], &[0.5f32; 32]);
    assert!((got - 8.0).abs() < 1e-3, "single-block dot = {got}");
}

// ---------------------------------------------------------------------------
// 7. documented limitation of the format (not a bug in our code):
//    d is stored in f16, min subnormal ~6e-8, so amax/127 below that
//    underflows to zero and the block reconstructs as all-zeros.
//    Real weights (~1e-2 scale) are nowhere near this cliff.
// ---------------------------------------------------------------------------

#[test]
fn f16_scale_underflow_is_a_real_format_limit() {
    let a = vec![1e-6f32; 32]; // d = 1e-6/127 ≈ 8e-9 < f16 min subnormal
    let b = vec![1.0f32; 32];

    let want = f32_dot(&a, &b);
    let got = quantized_dot(&a, &b);

    assert_ne!(want, 0.0);
    assert_eq!(got, 0.0, "expected scale underflow to flush the block to 0");
}

// ---------------------------------------------------------------------------
// 8. the AVX512F kernel, checked against the scalar fallback.
//
// The two implementations compute the per-block integer dot product exactly
// (i32, no overflow possible: 32 * 127 * 127 << 2^31) and only differ in the
// order of the f32 accumulation, so they must agree to f32 rounding noise —
// a tolerance of 1e-6 relative to the input scale is already generous.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use inference::q8_0_funcs::{q8_0_dot, q8_0_dot_avx512f};

    /// Skips the test instead of illegal-instructioning when the CPU is too old.
    ///
    /// The kernel needs AVX512F for the scales/`cvtph`, AVX512-VNNI for the
    /// `_mm256_dpbusd_epi32` accumulate and AVX512VL for the 128/256-bit
    /// forms — the same set `#[target_feature]` declares on the kernel.
    fn avx512_dot(a: &Q8_0, b: &Q8_0) -> Option<f32> {
        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512vnni")
            || !std::arch::is_x86_feature_detected!("avx512vl")
        {
            return None;
        }
        Some(unsafe { q8_0_dot_avx512f(a, b) })
    }

    /// The kernel and the fallback differ only in the order of the f32
    /// accumulation, so the error must be at f32 rounding-noise level.
    /// Normalized by `yardstick` (sum of |a_i*b_i|) and never by the dot
    /// itself: with mixed signs the dot nearly cancels and comparing against
    /// it would flag plain float reordering as an error.
    fn assert_matches_fallback(a: &Q8_0, b: &Q8_0, yardstick: f64, ctx: &str) {
        let Some(got) = avx512_dot(a, b) else {
            println!("skipping ({ctx}): no avx512f");
            return;
        };
        let fallback = q8_0_dot_fallback(a, b);
        let err = (got as f64 - fallback as f64).abs() / yardstick;
        assert!(
            err < 1e-6,
            "{ctx}: avx512 {got} vs fallback {fallback} (rel err {err})"
        );
    }

    /// The kernel vectorizes 16 blocks at a time and finishes with a
    /// one-block-at-a-time tail, so every side of that split must be covered.
    #[test]
    fn matches_fallback_across_chunk_boundaries() {
        let Some(_) = avx512_dot(&f32_to_q8_0(&[0.0; 32]), &f32_to_q8_0(&[0.0; 32])) else {
            println!("skipping: no avx512f + avx512_vnni");
            return;
        };

        for &blocks in &[1usize, 2, 15, 16, 17, 31, 32, 33, 128, 129] {
            let n = blocks * 32;
            for &scale in &[1e-2f32, 1.0, 1e2, 1e5] {
                let mut rng = Rng::new(0xDEAD_BEEF ^ (scale.to_bits() as u64) ^ n as u64);
                let a = rng.vec(n, scale);
                let b = rng.vec(n, scale);
                let qa = f32_to_q8_0(&a);
                let qb = f32_to_q8_0(&b);

                assert_matches_fallback(
                    &qa,
                    &qb,
                    l1_products(&a, &b),
                    &format!("blocks={blocks} scale={scale}"),
                );

                // and the kernel must be as close to the f32 reference as the
                // fallback is (quantization error, not kernel error)
                let want = f32_dot(&a, &b);
                let yardstick = l1_products(&a, &b);
                let got = avx512_dot(&qa, &qb).unwrap();
                assert!(
                    rel_err(got, want, yardstick) < 1e-2,
                    "blocks={blocks} scale={scale}: rel err vs f32 = {}",
                    rel_err(got, want, yardstick)
                );
            }
        }
    }

    /// Bit-exact agreement, not just "close enough": with dyadic inputs every
    /// intermediate (f16 scales, i32 block dots, f32 partial sums) is exactly
    /// representable, so both paths must produce identical f32 values and no
    /// rounding excuse exists. This is what actually pins down the integer
    /// part of the kernel — the sign trick, dpbusd and the per-block scale.
    #[test]
    fn is_bit_exact_on_dyadic_inputs() {
        let Some(_) = avx512_dot(&f32_to_q8_0(&[0.0; 32]), &f32_to_q8_0(&[0.0; 32])) else {
            println!("skipping: no avx512f + avx512_vnni");
            return;
        };

        // values are integers * 2^-10 with one element at exactly ±127*2^-10
        // per block, so d = 2^-10 exactly; |dot| per block <= 32*127*127 and
        // the total stays below 2^24 * 2^-20, i.e. every f32 sum is exact.
        for &blocks in &[1usize, 2, 15, 16, 17, 31] {
            let n = blocks * 32;
            let mut rng = Rng::new(0xB16 ^ n as u64);
            let mk = |rng: &mut Rng| {
                let mut v: Vec<f32> = (0..n)
                    .map(|_| (rng.next_u64() % 255) as i32 as f32 / 1024. - 127. / 1024.)
                    .collect();
                // pin amax of every block to exactly 127*2^-10 so d = 2^-10
                for k in 0..blocks {
                    v[k * 32 + 31] = 127. / 1024.;
                }
                v
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            let (qa, qb) = (f32_to_q8_0(&a), f32_to_q8_0(&b));

            let Some(got) = avx512_dot(&qa, &qb) else {
                println!("skipping: no avx512f + avx512_vnni");
                return;
            };
            let fallback = q8_0_dot_fallback(&qa, &qb);
            assert_eq!(got, fallback, "blocks={blocks}: not bit-exact");

            // and both must equal the unquantized f32 dot bit-for-bit too,
            // since these values quantize without any loss
            let want = f32_dot_fallback(&a, &b);
            assert_eq!(got, want, "blocks={blocks}: lost exactness vs f32");
        }
    }

    #[test]
    fn matches_fallback_on_special_shapes() {
        let mut rng = Rng::new(7);

        // zero block next to a real one: no NaN, nothing leaked
        let mut a = vec![0.0f32; 32];
        a.extend_from_slice(&rng.vec(32, 1.0));
        let mut b = vec![0.0f32; 32];
        b.extend_from_slice(&rng.vec(32, 1.0));
        let (qa, qb) = (f32_to_q8_0(&a), f32_to_q8_0(&b));
        let Some(got) = avx512_dot(&qa, &qb) else {
            println!("skipping: no avx512f + avx512_vnni");
            return;
        };
        assert!(!got.is_nan(), "avx512 leaked NaN: {got}");
        assert_matches_fallback(&qa, &qb, l1_products(&a, &b), "zero block");

        // outlier block (one huge value, rest tiny)
        let mut a = vec![1e-3f32; 32];
        a[0] = 1e3;
        let mut b = vec![1e-3f32; 32];
        b[0] = -1e3;
        let (qa, qb) = (f32_to_q8_0(&a), f32_to_q8_0(&b));
        assert_matches_fallback(&qa, &qb, l1_products(&a, &b), "outlier");

        // constant vectors: every block quantizes to the same weights
        let a = vec![0.5f32; 4096];
        let b = vec![-0.25f32; 4096];
        let (qa, qb) = (f32_to_q8_0(&a), f32_to_q8_0(&b));
        assert_matches_fallback(&qa, &qb, l1_products(&a, &b), "constant");
    }

    /// Negative weights are exactly what the `sign_epi8` trick exists for, so
    /// pin the mixed-sign case down with a hand-built vector.
    #[test]
    fn matches_fallback_on_mixed_signs() {
        // alternating signs, ramp magnitudes, repeated to fill many blocks
        let block: Vec<f32> = (0..32)
            .map(|j| if j % 2 == 0 { (j + 1) as f32 / 32. } else { -((j + 1) as f32 / 32.) })
            .collect();
        let a: Vec<f32> = block.iter().copied().cycle().take(32 * 40).collect();
        let b: Vec<f32> = block.iter().rev().copied().cycle().take(32 * 40).collect();
        let (qa, qb) = (f32_to_q8_0(&a), f32_to_q8_0(&b));
        assert_matches_fallback(&qa, &qb, l1_products(&a, &b), "mixed signs");
    }

    /// The dispatcher must actually route to the kernel on this machine.
    #[test]
    fn dispatcher_uses_the_kernel_when_available() {
        let Some(_) = avx512_dot(&f32_to_q8_0(&[0.0; 32]), &f32_to_q8_0(&[0.0; 32])) else {
            println!("skipping: no avx512f + avx512_vnni");
            return;
        };
        let mut rng = Rng::new(0x5EED);
        let a = rng.vec(4096, 1.0);
        let b = rng.vec(4096, 1.0);
        let (qa, qb) = (f32_to_q8_0(&a), f32_to_q8_0(&b));

        let dispatched = q8_0_dot(&qa, &qb);
        let kernel = unsafe { q8_0_dot_avx512f(&qa, &qb) };
        assert_eq!(
            dispatched, kernel,
            "q8_0_dot did not take the avx512 path"
        );
    }
}
