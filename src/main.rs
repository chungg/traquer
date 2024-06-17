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

    //dbg!(trend::adx(&stats.high, &stats.low, &stats.close, 14, 14).collect::<Vec<_>>());
    dbg!(momentum::tsi(&stats.close, 6, 10, 3).collect::<Vec<(f64, f64)>>());
    //dbg!(
    //    volatility::heikin_ashi(&stats.open, &stats.high, &stats.low, &stats.close)
    //        .collect::<Vec<_>>()
    //);
    //dbg!(smooth::alma(&stats.close, 10, 6.0, None).collect::<Vec<f64>>());
}
