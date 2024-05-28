use serde::{Deserialize, Serialize};
use std::fs;

use traquer::*;

#[derive(Deserialize, Serialize, Debug)]
struct SecStats {
    high: Vec<f64>,
    low: Vec<f64>,
    open: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

fn main() {
    let data = fs::read_to_string("./tests/rddt.input").expect("Unable to read file");
    let stats: SecStats = serde_json::from_str(&data).expect("JSON does not have correct format.");

    dbg!(momentum::pgo(&stats.high, &stats.low, &stats.close, 16));
    //dbg!(smooth::sma(&stats.close, 16).collect::<Vec<f64>>());
}
