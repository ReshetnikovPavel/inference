use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-link-lib=ze_loader");
    println!("cargo:rerun-if-changed=kernels/kernel.cl");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let status = Command::new("ocloc")
        .args([
            "compile",
            "-file",
            "kernels/kernel.cl",
            "-device",
            "tgl",
            "-out_dir",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute 'ocloc'. Ensure 'intel-compute-runtime' is installed.");

    assert!(status.success(), "ocloc failed to compile kernels/kernel.cl");
}
