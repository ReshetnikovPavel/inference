mod types;
mod ops;

use types::q8_0;

fn main() {
    let a = [1.; 2_usize.pow(12)];
    let b = [1.; 2_usize.pow(12)];
    let f32_res = ops::dot_f32(&a, &b);

    let a = q8_0::quantize(&a);
    let b = q8_0::quantize(&b);
    let q8_0_res = ops::dot_q8_0(&a, &b);

    dbg!(f32_res);
    dbg!(q8_0_res);
}
