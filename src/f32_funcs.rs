pub fn f32_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return unsafe { f32_dot_avx512f(a, b) };
        }
    }

    f32_dot_fallback(a, b)
}

pub fn f32_dot_fallback(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

#[target_feature(enable = "avx512f")]
unsafe fn f32_dot_avx512f(a: &[f32], b: &[f32]) -> f32 {
    const N: usize = 16;
    let mut acc = std::arch::x86_64::_mm512_setzero_ps();
    let mut i = 0;
    while i + N <= a.len() {
        unsafe {
            let chunk_a = std::arch::x86_64::_mm512_loadu_ps(a.as_ptr().add(i));
            let chunk_b = std::arch::x86_64::_mm512_loadu_ps(b.as_ptr().add(i));
            acc = std::arch::x86_64::_mm512_fmadd_ps(chunk_a, chunk_b, acc);
        }
        i += N;
    }

    let remaining = a.len() - i;
    if remaining > 0 {
        let mask = (1u16 << remaining) - 1;
        unsafe {
            let chunk_a = std::arch::x86_64::_mm512_maskz_loadu_ps(mask, a.as_ptr().add(i));
            let chunk_b = std::arch::x86_64::_mm512_maskz_loadu_ps(mask, b.as_ptr().add(i));
            acc = std::arch::x86_64::_mm512_fmadd_ps(chunk_a, chunk_b, acc);
        }
    }

    std::arch::x86_64::_mm512_reduce_add_ps(acc)
}

pub struct Mat<T> {
    data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Mat<T> {
    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

pub fn f32_linear(w: &Mat<f32>, x: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.cols, x.len());
    debug_assert_eq!(w.rows, b.len());

    let mut res = vec![0.; w.rows];

    for i in 0..w.rows {
        let row = w.row(i);
        res[i] = f32_dot(row, x) + b[i];
    }

    res
}

pub fn f32_relu(x: f32) -> f32 {
    x.max(0.)
}
