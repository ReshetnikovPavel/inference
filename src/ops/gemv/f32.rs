use crate::types::mat::Mat;
use crate::ops;

pub fn gemv(w: &Mat<f32>, x: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.cols, x.len());
    debug_assert_eq!(w.rows, b.len());

    let mut res = vec![0.; w.rows];

    for i in 0..w.rows {
        let row = w.row(i);
        res[i] = ops::dot_f32(row, x) + b[i];
    }

    res
}
