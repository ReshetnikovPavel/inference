use std::{
    fs::{self, File, OpenOptions},
    io::{Read as _, Write},
    path::Path,
    time::SystemTime,
};

pub fn gen_random_data(len: usize, n: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let total_bytes_needed = len * n * 2;
    let mut raw_bytes = vec![0u8; total_bytes_needed];
    let mut urandom = File::open("/dev/urandom").unwrap();

    urandom.read_exact(&mut raw_bytes).unwrap();

    let mut all_floats = raw_bytes.into_iter().map(|b| (b as i8 as f32) / 128.0);

    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let mut vec_a = Vec::with_capacity(len);
        let mut vec_b = Vec::with_capacity(len);

        for _ in 0..len {
            vec_a.push(all_floats.next().unwrap());
        }
        for _ in 0..len {
            vec_b.push(all_floats.next().unwrap());
        }

        data.push((vec_a, vec_b));
    }

    data
}

pub fn save_results(data_type: &str, lengths: &[usize], runs: usize, avgs: &[u128]) {
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

    let path = format!("logs/inference__dot__{data_type}.csv");
    let path = Path::new(&path);
    let existed = path.exists();

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();

    if !existed {
        file.write_all(
            format!(
                "engine,op,data_type,time,commit,has_untracted_changes,power_profile,length,runs,avg\n"
            )
            .as_bytes(),
        )
        .expect("Unable to write a header to a csv");
    }

    for (length, avg) in lengths.iter().zip(avgs) {
        file.write_all(format!("inference,dot,{data_type},{time},{commit},{has_untracted_changes},{power_profile},{length},{runs},{avg}\n").as_bytes()).expect("Unable to write a line to a csv");
    }
}
