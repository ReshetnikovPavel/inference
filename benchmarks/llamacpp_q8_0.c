// llama.cpp side of the Q8_0 dot-product comparison: quantize with llama.cpp's
// own quantizer, time llama.cpp's own kernel, print the same line format as
// benches/q8_0_vs_llamacpp.rs so the two outputs can be merged.
//
// Build & run (llama.cpp checked out at ~/opensource/llama.cpp):
//   LLAMA=~/opensource/llama.cpp
//   gcc -O3 -march=native -fPIC -I $LLAMA/ggml/include -I $LLAMA/ggml/src \
//       -I $LLAMA/ggml/src/ggml-cpu \
//       -c $LLAMA/ggml/src/ggml-cpu/arch/x86/quants.c -o /tmp/llamacpp_quants.o
//   gcc -O3 -march=native -fPIC -I $LLAMA/ggml/include -I $LLAMA/ggml/src \
//       -I $LLAMA/ggml/src/ggml-cpu \
//       -c $LLAMA/ggml/src/ggml-cpu/quants.c -o /tmp/llamacpp_quants_generic.o
//   gcc -O3 -march=native benchmarks/llamacpp_q8_0.c \
//       /tmp/llamacpp_quants.o /tmp/llamacpp_quants_generic.o \
//       -L $LLAMA/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$LLAMA/build/bin \
//       -lm -o /tmp/llamacpp_q8_0 && /tmp/llamacpp_q8_0
//
// The two .o files are llama.cpp's own sources compiled for *this* CPU on
// purpose: the prebuilt libggml-cpu.so in that build dir was compiled without
// VNNI, and llama.cpp's q8_0 kernel only ever had an AVX2 path anyway (see
// ggml_vec_dot_q8_0_q8_0 in arch/x86/quants.c: `#if defined(__AVX2__)`, no
// AVX512/VNNI variant). Both symbols end up defined in the executable (T), so
// the linked .so's cannot shadow them; the .so is only there for
// quantize_row_q8_0_ref and for the fp-e8m0/ue4m3 tables quants.c references.
//
// Everything that must line up with the Rust side:
//   - data: xorshift64 stream, seed = 0xC0FFEE ^ n, then `| 1`, exactly like
//     q8_0_pairs() in benches/ops.rs. Values are (x >> 40) / 2^24 * 2 - 1,
//     which is exact in both f32 and float, so both programs see identical bits.
//   - block layout: `uint16_t d; int8_t qs[32];` — identical to llama.cpp's
//     block_q8_0 (ggml_half d + 32 weights, no padding) and to Rust's
//     (F16, [i8; 32]).
//   - dot call: same arguments llama.cpp's own hot path uses
//     (bs = bx = by = 0, nrc = 1 — see ggml-cpu.c mul_mat).
//   - timing: 1000 warmup + 10000 timed iterations, mean per call.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// llama.cpp's block, spelled out to keep this file header-free.
typedef struct {
    uint16_t d;        // ggml_half scale
    int8_t   qs[32];   // QK8_0 = 32 weights
} block_q8_0;

void quantize_row_q8_0_ref(const float * x, block_q8_0 * y, int64_t k);
void ggml_vec_dot_q8_0_q8_0        (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static const int    ITERS  = 10000;
static const int    WARMUP = 1000;
static const int    ROUNDS = 7;
static const size_t SIZES[] = { 32, 128, 512, 544, 1024, 4096, 16384, 65536, 262144 };

// --- data generation, identical to q8_0_pairs() in benches/ops.rs -----------
static uint64_t rng_state;

static uint64_t next_u64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static void gen_f32(float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = (float)(next_u64() >> 40) / 16777216.0f * 2.0f - 1.0f;
    }
}

// --- FNV-1a over the quantized bytes, so the Rust side can prove it produced
//     bit-identical q8_0 blocks. Both operands are hashed into one stream.
static uint64_t quant_hash(const block_q8_0 * q, size_t nb, uint64_t h) {
    const unsigned char * b = (const unsigned char *) q;
    for (size_t i = 0; i < nb * sizeof(block_q8_0); i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// Interleaved timing, matching the Rust side: every implementation is
// measured once per round on the same data and the best round is reported,
// so turbo/thermal drift hits both implementations inside the same round.
static void bench_all(int n, const void * x, const void * y,
                      const char * const * names, const dot_fn * fs,
                      const int * inners, size_t count) {
    float value[8];
    double best[8];

    for (size_t k = 0; k < count; k++) {
        float s = 0.0f;
        for (int i = 0; i < WARMUP; i++) {
            fs[k](n, &s, 0, x, 0, y, 0, 1);
        }
        best[k] = 1e30;
        value[k] = 0.0f;
    }

    for (int round = 0; round < ROUNDS; round++) {
        for (size_t k = 0; k < count; k++) {
            float s = 0.0f;
            const double t0 = now_sec();
            for (int i = 0; i < inners[k]; i++) {
                fs[k](n, &s, 0, x, 0, y, 0, 1);
            }
            const double dt = (now_sec() - t0) * 1e9 / inners[k];
            // keep the result observable so nothing is elided
            if (s == 12345.6789f) {
                printf("unreachable %s\n", names[k]);
            }
            if (dt < best[k]) {
                best[k] = dt;
            }
            value[k] = s;
        }
    }

    for (size_t k = 0; k < count; k++) {
        printf("RES n=%d impl=%s value=%.9g ns=%.2f gpairs=%.2f\n",
               n, names[k], (double) value[k], best[k], (double) n / best[k]);
    }
}

int main(void) {
    printf("# llama.cpp q8_0: quantize_row_q8_0_ref + ggml_vec_dot_q8_0_q8_0\n");
    printf("# n impl quant_hash value ns gpairs\n");

    for (size_t si = 0; si < sizeof(SIZES)/sizeof(SIZES[0]); si++) {
        const size_t n = SIZES[si];
        const size_t nb = n / 32;

        rng_state = (0xC0FFEE ^ n) | 1;
        float * a = malloc(n * sizeof(float));
        float * b = malloc(n * sizeof(float));
        gen_f32(a, n);
        gen_f32(b, n);

        // f64 reference over the same data, same summation order as Rust
        double ref = 0.0;
        for (size_t i = 0; i < n; i++) {
            ref += (double) a[i] * (double) b[i];
        }

        block_q8_0 * qa = malloc(nb * sizeof(block_q8_0));
        block_q8_0 * qb = malloc(nb * sizeof(block_q8_0));
        quantize_row_q8_0_ref(a, qa, (int64_t) n);
        quantize_row_q8_0_ref(b, qb, (int64_t) n);

        const uint64_t hash = quant_hash(qb, nb, quant_hash(qa, nb, 14695981039346656037ULL));
        printf("REF n=%zu ref=%.17g hash=%016llx\n", n, ref, (unsigned long long) hash);

        const char * names[] = { "llamacpp_avx2" };
        const dot_fn fs[] = { ggml_vec_dot_q8_0_q8_0 };
        const int inners[] = { ITERS };
        bench_all((int) n, qa, qb, names, fs, inners, 1);

        // the scalar generic path is ~2x slower per element and many times
        // slower overall; measuring it in the same interleaved pass would
        // evict the AVX2 kernel's working set and inflate its best time, so
        // it gets its own pass.
        const char * gnames[] = { "llamacpp_generic" };
        const dot_fn gfs[] = { ggml_vec_dot_q8_0_q8_0_generic };
        const int ginners[] = { ITERS / 100 };
        bench_all((int) n, qa, qb, gnames, gfs, ginners, 1);

        free(a); free(b); free(qa); free(qb);
    }
    return 0;
}
