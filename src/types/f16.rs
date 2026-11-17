use std::arch::x86_64;

#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct F16(i16);

impl From<f32> for F16 {
    #[inline]
    fn from(value: f32) -> Self {
        unsafe { f16_from_f32_f16c(value) }
    }
}

impl From<F16> for f32 {
    #[inline]
    fn from(value: F16) -> f32 {
        unsafe { f32_from_f16_f16c(value) }
    }
}

#[inline]
#[target_feature(enable = "f16c")]
unsafe fn f16_from_f32_f16c(x: f32) -> F16 {
    let padded_f32 = [x, 0., 0., 0.];
    let padded_f32 = unsafe { x86_64::_mm_loadu_ps(padded_f32.as_ptr()) };
    let padded_f16 = x86_64::_mm_cvtps_ph::<{ x86_64::_MM_FROUND_TO_NEAREST_INT }>(padded_f32);
    let res = x86_64::_mm_extract_epi16::<0>(padded_f16) as i16;
    F16(res)
}

#[inline]
#[target_feature(enable = "f16c")]
unsafe fn f32_from_f16_f16c(x: F16) -> f32 {
    let padded_f16 = x86_64::_mm_cvtsi32_si128(x.0 as i32);
    let padded_f32 = x86_64::_mm_cvtph_ps(padded_f16);
    x86_64::_mm_cvtss_f32(padded_f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: F16) -> u16 {
        value.0 as u16
    }

    #[test]
    fn layout_is_two_bytes() {
        assert_eq!(size_of::<F16>(), 2);
        assert_eq!(align_of::<F16>(), 2);
    }

    #[test]
    fn default_is_positive_zero() {
        assert_eq!(bits(F16::default()), 0x0000);
        assert!(f32::from(F16::default()).is_sign_positive());
    }

    #[test]
    fn known_values_encode() {
        let cases = [
            (0.0f32, 0x0000u16),
            (f32::from_bits(0x8000_0000), 0x8000),
            (0.25, 0x3400),
            (0.5, 0x3800),
            (1.0, 0x3C00),
            (-1.0, 0xBC00),
            (2.0, 0x4000),
            (-2.5, 0xC100),
            (1024.0, 0x6400),
            (0.1, 0x2E66),
            (1.0 / 3.0, 0x3555),
            (65504.0, 0x7BFF),
        ];
        for (value, expected) in cases {
            assert_eq!(bits(F16::from(value)), expected, "encoding {value}");
        }
    }

    #[test]
    fn known_values_decode() {
        let cases = [
            (0x0000u16, 0.0f32),
            (0x3400, 0.25),
            (0x3800, 0.5),
            (0x3C00, 1.0),
            (0xBC00, -1.0),
            (0x4000, 2.0),
            (0xC100, -2.5),
            (0x2E66, 1638.0 / 16384.0),
            (0x3555, 1365.0 / 4096.0),
            (0x7BFF, 65504.0),
            (0x0001, 2f32.powi(-24)),
            (0x03FF, 1023.0 * 2f32.powi(-24)),
            (0x0400, 2f32.powi(-14)),
            (0x7C00, f32::INFINITY),
            (0xFC00, f32::NEG_INFINITY),
        ];
        for (raw, expected) in cases {
            assert_eq!(f32::from(F16(raw as i16)), expected, "decoding {raw:#06X}");
        }
        assert_eq!(f32::from(F16(0x8000u16 as i16)).to_bits(), 0x8000_0000);
    }

    #[test]
    fn overflow_rounds_to_infinity() {
        let cases = [
            (65519.0f32, 0x7BFFu16),
            (65520.0, 0x7C00),
            (65536.0, 0x7C00),
            (1.0e10, 0x7C00),
            (f32::MAX, 0x7C00),
            (-65519.0, 0xFBFF),
            (-65520.0, 0xFC00),
            (f32::MIN, 0xFC00),
        ];
        for (value, expected) in cases {
            assert_eq!(bits(F16::from(value)), expected, "encoding {value}");
        }
    }

    #[test]
    fn underflow_rounds_to_zero() {
        let tiny = 2f32.powi(-24);
        let half = 2f32.powi(-25);
        let cases = [
            (tiny, 0x0001u16),
            (-tiny, 0x8001),
            (half, 0x0000),
            (-half, 0x8000),
            (half * 0.5, 0x0000),
            (half * 0.999, 0x0000),
            (half * 1.5, 0x0001),
        ];
        for (value, expected) in cases {
            assert_eq!(bits(F16::from(value)), expected, "encoding {value:e}");
        }
    }

    #[test]
    fn ties_round_to_even() {
        let step = 2f32.powi(-11);
        assert_eq!(bits(F16::from(1.0 + step)), 0x3C00);
        assert_eq!(bits(F16::from(1.0 + 2.0 * step)), 0x3C01);
        assert_eq!(bits(F16::from(1.0 + 3.0 * step)), 0x3C02);

        let step = 2f32.powi(-10);
        assert_eq!(bits(F16::from(2.0 + step)), 0x4000);
        assert_eq!(bits(F16::from(2.0 + 3.0 * step)), 0x4002);
    }

    #[test]
    fn subnormal_boundary_rounds() {
        let smallest_normal = 2f32.powi(-14);
        let largest_subnormal = (1.0 - 2f32.powi(-10)) * smallest_normal;
        let midpoint = (smallest_normal + largest_subnormal) / 2.0;

        assert_eq!(bits(F16::from(smallest_normal)), 0x0400);
        assert_eq!(bits(F16::from(largest_subnormal)), 0x03FF);
        assert_eq!(bits(F16::from(midpoint)), 0x0400);
        assert_eq!(bits(F16::from(midpoint * (1.0 - 2f32.powi(-20)))), 0x03FF);
        assert_eq!(bits(F16::from(midpoint * (1.0 + 2f32.powi(-20)))), 0x0400);
    }

    #[test]
    fn special_values() {
        assert_eq!(bits(F16::from(f32::INFINITY)), 0x7C00);
        assert_eq!(bits(F16::from(f32::NEG_INFINITY)), 0xFC00);

        let inputs = [
            f32::NAN,
            f32::from_bits(0x7F80_0001),
            f32::from_bits(0xFF80_0001),
            f32::from_bits(0x7FC0_0000),
        ];
        for value in inputs {
            let raw = bits(F16::from(value));
            assert_eq!(raw & 0x7C00, 0x7C00, "{value} lost the all-ones exponent");
            assert_ne!(raw & 0x03FF, 0, "{value} became infinity");
            assert!(f32::from(F16(raw as i16)).is_nan());
        }

        for raw in [0x7C01u16, 0x7E00, 0xFE00, 0xFFFF] {
            assert!(f32::from(F16(raw as i16)).is_nan(), "0x{raw:04X} decoded as a number");
        }
    }

    #[test]
    fn roundtrip_all_bit_patterns() {
        for raw in 0..=u16::MAX {
            let decoded = f32::from(F16(raw as i16));
            if decoded.is_nan() {
                assert!(
                    f32::from(F16::from(decoded)).is_nan(),
                    "NaN 0x{raw:04X} did not round-trip as NaN"
                );
            } else {
                assert_eq!(
                    bits(F16::from(decoded)),
                    raw,
                    "0x{raw:04X} did not round-trip"
                );
            }
        }
    }
}
