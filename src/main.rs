mod f32_funcs;
mod q8_0_funcs;

fn main() {
    let a = [1.; 2_usize.pow(12)];
    let b = [1.; 2_usize.pow(12)];
    let f32_res = f32_funcs::f32_dot(&a, &b);

    let a = q8_0_funcs::f32_to_q8_0(&a);
    let b = q8_0_funcs::f32_to_q8_0(&b);
    let q8_0_res = q8_0_funcs::q8_0_dot_fallback(&a, &b);

    dbg!(f32_res);
    dbg!(q8_0_res);
}
