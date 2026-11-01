use std::{
    fs::{self, File, OpenOptions},
    io::{Read as _, Write},
    path::Path,
    time::SystemTime,
};

use inference::types::mat::Mat;

pub fn gen_random_data(m: usize, n: usize, runs: usize) -> (Mat<f32>, Vec<(Vec<f32>, Vec<f32>)>) {
    let total_bytes_needed = (m * n) + ((n + m) * runs);
    let mut raw_bytes = vec![0u8; total_bytes_needed];
    let mut urandom = File::open("/dev/urandom").unwrap();
    urandom.read_exact(&mut raw_bytes).unwrap();

    let mut all_floats = raw_bytes.into_iter().map(|b| (b as i8 as f32) / 128.0);

    let mut mat_data = Vec::with_capacity(m * n);
    for _ in 0..(m * n) {
        mat_data.push(all_floats.next().unwrap());
    }
    let w = Mat::new(mat_data, m, n);

    let mut vectors = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut vec_x = Vec::with_capacity(n);
        let mut vec_b = Vec::with_capacity(m);

        for _ in 0..n {
            vec_x.push(all_floats.next().unwrap());
        }
        for _ in 0..m {
            vec_b.push(all_floats.next().unwrap());
        }

        vectors.push((vec_x, vec_b));
    }

    (w, vectors)
}

pub fn save_results_gemv(
    data_type: &str,
    dimensions: &[(usize, usize)],
    runs: usize,
    avgs: &[u128],
) {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .expect("Unable to read current git commit");
    assert!(commit.status.success());
    let commit = String::from_utf8_lossy(&commit.stdout);
    let commit = commit.trim();

    let has_untracted_changes = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .expect("Unable to get git status");
    assert!(has_untracted_changes.status.success());
    let has_untracted_changes = !has_untracted_changes.stdout.is_empty();

    let power_profile =
        fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
            .unwrap_or_default();
    let power_profile = power_profile.trim();

    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let path = format!("logs/inference__gemv__{data_type}.csv");
    let path = Path::new(&path);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let existed = path.exists();

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();

    if !existed {
        file.write_all(
            format!(
                "engine,op,data_type,time,commit,has_untracted_changes,power_profile,m,n,runs,avg\n"
            )
            .as_bytes(),
        )
        .expect("Unable to write a header to a csv");
    }

    for ((m, n), avg) in dimensions.iter().zip(avgs) {
        file.write_all(format!("inference,gemv,{data_type},{time},{commit},{has_untracted_changes},{power_profile},{m},{n},{runs},{avg}\n").as_bytes()).expect("Unable to write a line to a csv");
    }
}
