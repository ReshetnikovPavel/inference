use crate::types::{avec::CacheAlignedVec, f16::F16};

pub const WEIGHTS_IN_BLOCK: usize = 32;

pub struct VecQ8_0 {
    d: CacheAlignedVec<F16>,
    weights: CacheAlignedVec<i8>,
}

impl VecQ8_0 {
    #[inline]
    pub fn weights(&self) -> &[i8] {
        &self.weights
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
    let mut q8_0_d = CacheAlignedVec::new();
    let mut q8_0_weights = CacheAlignedVec::new();
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
        weights: q8_0_weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        let tolerance = expected.abs() * 2f32.powi(-11);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} is not within {tolerance} of {expected}"
        );
    }

    fn pseudo_random(len: usize) -> Vec<f32> {
        let mut state = 12345u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7FFF_FFFF;
                (state as f32 / 2147483647.0) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn empty_input_produces_no_blocks() {
        let v = quantize(&[]);
        assert_eq!(v.blocks_len(), 0);
        assert!(v.weights().is_empty());
        assert!(v.d().is_empty());
    }

    #[test]
    #[should_panic]
    fn partial_block_is_rejected() {
        let _ = quantize(&[0.0f32; WEIGHTS_IN_BLOCK + 1]);
    }

    #[test]
    fn lengths_match_input() {
        for len in [32, 64, 96, 1024] {
            let v = quantize(&vec![0.5f32; len]);
            assert_eq!(v.weights().len(), len);
            assert_eq!(v.d().len(), len / WEIGHTS_IN_BLOCK);
            assert_eq!(v.blocks_len(), len / WEIGHTS_IN_BLOCK);
        }
    }

    #[test]
    fn weights_block_matches_flat_weights() {
        let mut input = Vec::new();
        for block in 0..4 {
            for j in 0..WEIGHTS_IN_BLOCK {
                input.push(((block * 37 + j * 11) % 41) as f32 / 41.0 - 0.5);
            }
        }

        let v = quantize(&input);
        assert_eq!(v.blocks_len(), 4);

        let mut blocks = Vec::new();
        for i in 0..v.blocks_len() {
            let block: &[i8] = v.weights_block(i);
            assert_eq!(block.len(), WEIGHTS_IN_BLOCK);
            assert_eq!(
                block,
                &v.weights()[i * WEIGHTS_IN_BLOCK..(i + 1) * WEIGHTS_IN_BLOCK]
            );
            blocks.push(block.to_vec());
        }
        for i in 1..blocks.len() {
            assert_ne!(blocks[i], blocks[i - 1]);
        }
    }

    #[test]
    fn positive_peak_maps_to_full_scale() {
        let mut input = [0.0f32; WEIGHTS_IN_BLOCK];
        input[0] = 0.5;
        input[1] = -0.25;

        let v = quantize(&input);

        assert_eq!(v.weights()[0], 127);
        assert_close(f32::from(v.d()[0]), 0.5 / 127.0);
    }

    #[test]
    fn negative_peak_maps_to_negative_full_scale() {
        let mut input = [0.0f32; WEIGHTS_IN_BLOCK];
        input[0] = -3.0;
        input[1] = 1.0;

        let v = quantize(&input);

        assert_eq!(v.weights()[0], -127);
        assert_close(f32::from(v.d()[0]), 3.0 / 127.0);
    }

    #[test]
    fn zero_block_quantizes_to_zero() {
        let v = quantize(&[0.0f32; WEIGHTS_IN_BLOCK]);

        assert_eq!(v.d()[0].0, 0);
        assert!(f32::from(v.d()[0]).is_sign_positive());
        assert!(v.weights().iter().all(|&w| w == 0));
    }

    #[test]
    fn constant_block_saturates() {
        let positive = quantize(&[2.0f32; WEIGHTS_IN_BLOCK]);
        assert!(positive.weights().iter().all(|&w| w == 127));
        assert_close(f32::from(positive.d()[0]), 2.0 / 127.0);

        let negative = quantize(&[-2.0f32; WEIGHTS_IN_BLOCK]);
        assert!(negative.weights().iter().all(|&w| w == -127));
        assert_close(f32::from(negative.d()[0]), 2.0 / 127.0);
    }

    #[test]
    fn reconstruction_within_half_step() {
        let input = pseudo_random(1024);
        let v = quantize(&input);

        for block in 0..v.blocks_len() {
            let d = f32::from(v.d()[block]);
            let weights = v.weights_block(block);
            for j in 0..WEIGHTS_IN_BLOCK {
                let reconstructed = weights[j] as f32 * d;
                let original = input[block * WEIGHTS_IN_BLOCK + j];
                assert!(
                    (reconstructed - original).abs() <= 0.6 * d,
                    "block {block} element {j}: {original} reconstructed as {reconstructed} from d={d}"
                );
            }
        }
    }

    #[test]
    fn scale_matches_max_abs_over_127() {
        let input = pseudo_random(1024);
        let v = quantize(&input);

        for block in 0..v.blocks_len() {
            let values = &input[block * WEIGHTS_IN_BLOCK..(block + 1) * WEIGHTS_IN_BLOCK];
            let max_abs = values.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
            assert_close(f32::from(v.d()[block]), max_abs / 127.0);
        }
    }
}
