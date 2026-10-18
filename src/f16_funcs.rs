#[derive(Clone, Copy)]
pub struct F16(i16);

pub fn f16_from_f32(x: f32) -> F16 {
    unsafe { f16_from_f32_f16c(x) }
}

#[target_feature(enable = "f16c")]
unsafe fn f16_from_f32_f16c(x: f32) -> F16 {
    let padded_x_f32 = [x, 0., 0., 0.];
    let padded_x_f32 = unsafe { std::arch::x86_64::_mm_loadu_ps(padded_x_f32.as_ptr()) };
    let padded_x_f16 = std::arch::x86_64::_mm_cvtps_ph::<
        { std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT },
    >(padded_x_f32);
    let res = std::arch::x86_64::_mm_extract_epi16::<0>(padded_x_f16) as i16;
    F16(res)
}

pub fn f32_from_f16(x: F16) -> f32 {
    unsafe { f32_from_f16_f16c(x) }
}

#[target_feature(enable = "f16c")]
unsafe fn f32_from_f16_f16c(x: F16) -> f32 {
    let padded_x_f16 = [x.0, 0, 0, 0, 0, 0, 0, 0];
    let padded_x_f16 = unsafe { std::arch::x86_64::_mm_loadu_epi16(padded_x_f16.as_ptr()) };
    let padded_x_f32 = std::arch::x86_64::_mm_cvtph_ps(padded_x_f16);
    let res = std::arch::x86_64::_mm_extract_ps::<0>(padded_x_f32);
    let res = f32::from_bits(res as u32);
    res
}
