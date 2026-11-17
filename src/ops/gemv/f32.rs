use std::arch::x86_64;

use crate::ops;
use crate::types::mat::Mat;

pub fn gemv(w: &Mat<f32>, x: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.cols, x.len());
    debug_assert_eq!(w.rows, b.len());

    if is_x86_feature_detected!("avx512f") {
        return unsafe { gemv_avx512f(w, x, b) };
    }
    gemv_fallback(w, x, b)
}

fn gemv_fallback(w: &Mat<f32>, x: &[f32], b: &[f32]) -> Vec<f32> {
    let mut res = vec![0.; w.rows];

    for i in 0..w.rows {
        let row = w.row(i);
        res[i] = ops::dot_f32(row, x) + b[i];
    }

    res
}

#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_avx512f(w: &Mat<f32>, x: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.cols, x.len());
    debug_assert_eq!(w.rows, b.len());

    const N: usize = 16;
    let mut y = vec![0.0; w.rows];
    if w.rows == 0 {
        return y;
    }

    let cols = w.cols;
    let x_ptr = x.as_ptr();

    let mut base_w_ptr = w.row(0).as_ptr();

    let mut i = 0;
    while i + 4 <= w.rows {
        let row0 = base_w_ptr;
        let row1 = unsafe { base_w_ptr.add(cols) };
        let row2 = unsafe { base_w_ptr.add(cols * 2) };
        let row3 = unsafe { base_w_ptr.add(cols * 3) };

        let mut acc0 = x86_64::_mm512_setzero_ps();
        let mut acc1 = x86_64::_mm512_setzero_ps();
        let mut acc2 = x86_64::_mm512_setzero_ps();
        let mut acc3 = x86_64::_mm512_setzero_ps();

        let mut j = 0;
        while j + N <= cols {
            let chunk_x = unsafe { x86_64::_mm512_loadu_ps(x_ptr.add(j)) };

            let chunk_w0 = unsafe { x86_64::_mm512_loadu_ps(row0.add(j)) };
            acc0 = x86_64::_mm512_fmadd_ps(chunk_w0, chunk_x, acc0);

            let chunk_w1 = unsafe { x86_64::_mm512_loadu_ps(row1.add(j)) };
            acc1 = x86_64::_mm512_fmadd_ps(chunk_w1, chunk_x, acc1);

            let chunk_w2 = unsafe { x86_64::_mm512_loadu_ps(row2.add(j)) };
            acc2 = x86_64::_mm512_fmadd_ps(chunk_w2, chunk_x, acc2);

            let chunk_w3 = unsafe { x86_64::_mm512_loadu_ps(row3.add(j)) };
            acc3 = x86_64::_mm512_fmadd_ps(chunk_w3, chunk_x, acc3);

            j += N;
        }

        let remaining = cols - j;
        if remaining > 0 {
            let mask = (1u16 << remaining) - 1;
            let chunk_x = unsafe { x86_64::_mm512_maskz_loadu_ps(mask, x_ptr.add(j)) };

            let chunk_w0 = unsafe { x86_64::_mm512_maskz_loadu_ps(mask, row0.add(j)) };
            acc0 = x86_64::_mm512_fmadd_ps(chunk_w0, chunk_x, acc0);

            let chunk_w1 = unsafe { x86_64::_mm512_maskz_loadu_ps(mask, row1.add(j)) };
            acc1 = x86_64::_mm512_fmadd_ps(chunk_w1, chunk_x, acc1);

            let chunk_w2 = unsafe { x86_64::_mm512_maskz_loadu_ps(mask, row2.add(j)) };
            acc2 = x86_64::_mm512_fmadd_ps(chunk_w2, chunk_x, acc2);

            let chunk_w3 = unsafe { x86_64::_mm512_maskz_loadu_ps(mask, row3.add(j)) };
            acc3 = x86_64::_mm512_fmadd_ps(chunk_w3, chunk_x, acc3);
        }

        y[i] = x86_64::_mm512_reduce_add_ps(acc0) + b[i];
        y[i + 1] = x86_64::_mm512_reduce_add_ps(acc1) + b[i + 1];
        y[i + 2] = x86_64::_mm512_reduce_add_ps(acc2) + b[i + 2];
        y[i + 3] = x86_64::_mm512_reduce_add_ps(acc3) + b[i + 3];

        base_w_ptr = unsafe { base_w_ptr.add(cols * 4) };
        i += 4;
    }

    while i < w.rows {
        let row = base_w_ptr;
        let mut acc = x86_64::_mm512_setzero_ps();
        let mut j = 0;

        while j + N <= cols {
            let chunk_x = unsafe { x86_64::_mm512_loadu_ps(x_ptr.add(j)) };
            let chunk_w = unsafe { x86_64::_mm512_loadu_ps(row.add(j)) };
            acc = x86_64::_mm512_fmadd_ps(chunk_w, chunk_x, acc);
            j += N;
        }

        let remaining = cols - j;
        if remaining > 0 {
            let mask = (1u16 << remaining) - 1;
            let chunk_x = unsafe { x86_64::_mm512_maskz_loadu_ps(mask, x_ptr.add(j)) };
            let chunk_w = unsafe { x86_64::_mm512_maskz_loadu_ps(mask, row.add(j)) };
            acc = x86_64::_mm512_fmadd_ps(chunk_w, chunk_x, acc);
        }

        y[i] = x86_64::_mm512_reduce_add_ps(acc) + b[i];
        base_w_ptr = unsafe { base_w_ptr.add(cols) };
        i += 1;
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_i8(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7FFF_FFFF;
                (state % 17) as f32 - 8.0
            })
            .collect()
    }

    fn expected_gemv(w: &[f32], rows: usize, cols: usize, x: &[f32], b: &[f32]) -> Vec<f32> {
        (0..rows)
            .map(|i| {
                let row = &w[i * cols..(i + 1) * cols];
                let sum: i64 = row
                    .iter()
                    .zip(x)
                    .map(|(&w, &x)| w as i64 * x as i64)
                    .sum();
                (sum + b[i] as i64) as f32
            })
            .collect()
    }

    #[test]
    fn known_values() {
        let w = Mat::new(vec![1.0, 2.0, 3.0], 1, 3);
        let x = [4.0, 5.0, 6.0];
        let b = [7.0];
        let expected = [39.0];

        assert_eq!(gemv_fallback(&w, &x, &b), expected);
        assert_eq!(unsafe { gemv_avx512f(&w, &x, &b) }, expected);
        assert_eq!(gemv(&w, &x, &b), expected);
    }

    #[test]
    fn zero_rows_zero_cols_is_empty() {
        let w = Mat::new(vec![], 0, 0);
        let x = [];
        let b = [];
        let expected: Vec<f32> = vec![];

        assert_eq!(gemv_fallback(&w, &x, &b), expected);
        assert_eq!(unsafe { gemv_avx512f(&w, &x, &b) }, expected);
        assert_eq!(gemv(&w, &x, &b), expected);
    }

    #[test]
    fn zero_rows_with_cols_is_empty() {
        let w = Mat::new(vec![], 0, 3);
        let x = [0.5, 0.5, 0.5];
        let b = [];
        let expected: Vec<f32> = vec![];

        assert_eq!(gemv_fallback(&w, &x, &b), expected);
        assert_eq!(unsafe { gemv_avx512f(&w, &x, &b) }, expected);
        assert_eq!(gemv(&w, &x, &b), expected);
    }

    #[test]
    fn zero_cols_returns_bias() {
        let w = Mat::new(vec![], 5, 0);
        let x = [];
        let b = [1.0, -2.0, 3.0, -4.0, 5.0];
        let expected = [1.0, -2.0, 3.0, -4.0, 5.0];

        assert_eq!(gemv_fallback(&w, &x, &b), expected);
        assert_eq!(unsafe { gemv_avx512f(&w, &x, &b) }, expected);
        assert_eq!(gemv(&w, &x, &b), expected);
    }

    #[test]
    fn all_dimensions_match_expected() {
        for rows in 1..=9 {
            for cols in [0, 1, 15, 16, 17, 31, 32, 64, 1024] {
                let w = pseudo_random_i8(rows * cols, 1);
                let x = pseudo_random_i8(cols, 2);
                let b = pseudo_random_i8(rows, 3);
                let expected = expected_gemv(&w, rows, cols, &x, &b);
                let w = Mat::new(w, rows, cols);

                assert_eq!(
                    gemv_fallback(&w, &x, &b),
                    expected,
                    "rows {rows} cols {cols}"
                );
                assert_eq!(
                    unsafe { gemv_avx512f(&w, &x, &b) },
                    expected,
                    "rows {rows} cols {cols}"
                );
                assert_eq!(gemv(&w, &x, &b), expected, "rows {rows} cols {cols}");
            }
        }
    }
}
