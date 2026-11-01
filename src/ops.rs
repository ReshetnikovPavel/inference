use crate::types::{mat::Mat, q8_0::VecQ8_0};

mod dot;
mod gemv;

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    dot::f32::dot(a, b)
}

#[inline]
pub fn dot_q8_0(a: &VecQ8_0, b: &VecQ8_0) -> f32 {
    dot::q8_0::dot(a, b)
}

#[inline]
pub fn gemv_f32(w: &Mat<f32>, x: &[f32], b: &[f32]) -> Vec<f32> {
    gemv::f32::gemv(w, x, b)
}
