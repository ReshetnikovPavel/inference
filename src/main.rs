use inference::f32_dot;

fn main() {
    let a = [1.; 1000];
    let b = [1.; 1000];
    let res = f32_dot(&a, &b);
    let _ = res;
}
