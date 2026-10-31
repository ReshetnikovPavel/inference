use std::arch::x86_64;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct F16(pub i16);

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
