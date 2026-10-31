use crate::types::{cache::CacheAlignedVecI8, f16::F16};

pub const WEIGHTS_IN_BLOCK: usize = 32;

pub struct VecQ8_0 {
    d: Vec<F16>,
    weights: CacheAlignedVecI8,
}

impl VecQ8_0 {
    #[inline]
    pub fn weights(&self) -> &[i8] {
        &self.weights.data()
    }

    #[inline]
    pub fn weights_block(&self, i: usize) -> &[i8] {
        debug_assert!(i < self.d.len());
        unsafe {
            self.weights()[i * WEIGHTS_IN_BLOCK..(i + 1) * WEIGHTS_IN_BLOCK]
                .as_ptr()
                .cast::<[i8; WEIGHTS_IN_BLOCK]>()
                .as_ref_unchecked()
        }
    }

    #[inline]
    pub fn d(&self) -> &[F16] {
        &self.d
    }

    #[inline]
    pub fn blocks_len(&self) -> usize {
        self.d.len()
    }
}

pub fn quantize(v: &[f32]) -> VecQ8_0 {
    assert_eq!(v.len() % WEIGHTS_IN_BLOCK, 0);
    let mut q8_0_d = vec![];
    let mut q8_0_weights = CacheAlignedVecI8::new();
    let mut i = 0;
    while i + WEIGHTS_IN_BLOCK <= v.len() {
        let mut weights = [0; WEIGHTS_IN_BLOCK];
        let mut d = 0.;
        for j in 0..WEIGHTS_IN_BLOCK {
            let abs = v[i + j].abs();
            if abs > d {
                d = abs;
            }
        }
        d /= 127.;
        let id = if d == 0. { 0. } else { 1. / d };

        for j in 0..WEIGHTS_IN_BLOCK {
            weights[j] = (v[i + j] * id).round() as i8;
        }
        q8_0_weights.extend_from_slice(&weights);
        q8_0_d.push(d.into());

        i += WEIGHTS_IN_BLOCK;
    }

    VecQ8_0 {
        d: q8_0_d,
        weights: q8_0_weights.into(),
    }
}
