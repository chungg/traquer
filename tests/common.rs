use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Deserialize, Serialize, Debug)]
pub struct SecStats {
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub open: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

pub fn test_data() -> SecStats {
    test_data_path("./tests/rddt.input")
}

pub fn test_data_path(path: &str) -> SecStats {
    let data = fs::read_to_string(path).expect("Unable to read file");
    let stats: SecStats = serde_json::from_str(&data).expect("JSON does not have correct format.");
    stats
}

#[allow(dead_code)]
pub fn vec_eq(v1: &[f64], v2: &[f64]) -> bool {
    ((v1.len() == v2.len())
        && v1
            .iter()
            .zip(v2)
            .all(|(x, y)| (x.is_nan() && y.is_nan()) || (x == y)))
        || panic!(
            "assertion `left == right` failed\n left: {:?},\n right: {:?}",
            v1, v2
        )
}

#[allow(dead_code)]
pub fn vec_close(v1: &[f64], v2: &[f64]) -> bool {
    ((v1.len() == v2.len())
        && v1
            .iter()
            .zip(v2)
            .all(|(x, y)| (x.is_nan() && y.is_nan()) || ((x - y).abs() < 1e-8)))
        || panic!(
            "assertion `left == right` failed\n left: {:?},\n right: {:?}",
            v1, v2
        )
}
