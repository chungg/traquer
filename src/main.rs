use serde::{Deserialize, Serialize};
use std::fs;

use traquer::{indicator, smooth};

#[derive(Deserialize, Serialize, Debug)]
struct SecStats {
    high: Vec<f64>,
    low: Vec<f64>,
    open: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

fn main() {
    let data = fs::read_to_string("./rddt.input").expect("Unable to read file");

    let stats: SecStats = serde_json::from_str(&data).expect("JSON does not have correct format.");

    let _ = indicator::klinger(&stats.high, &stats.low, &stats.close, &stats.volume, 13, 21);
    let _ = indicator::klinger_vol(&stats.high, &stats.low, &stats.close, &stats.volume, 13, 21);
    let _ = indicator::twiggs(&stats.high, &stats.low, &stats.close, &stats.volume, 21);
    let _ = indicator::shinohara(&stats.high, &stats.low, &stats.close, 26);
    let _ = indicator::rsi(&stats.close, 14);
    let _ = indicator::macd(&stats.close, 12, 26);
    let _ = indicator::cmo(&stats.close, 10);
    let _ = indicator::cog(&stats.close, 10);
    let _ = indicator::acc_dist(&stats.high, &stats.low, &stats.close, &stats.volume);
    let _ = indicator::acc_dist_yahoo(&stats.high, &stats.low, &stats.close, &stats.volume);
    let _ = indicator::elder_ray(&stats.high, &stats.low, &stats.close, 13);
    let _ = indicator::elder_force(&stats.close, &stats.volume, 13);
    let _ = indicator::mfi(&stats.high, &stats.low, &stats.close, &stats.volume, 13);

    let _ = smooth::vma(&stats.close, 10);
    let _ = smooth::vidya(&stats.close, 10);
    let _ = smooth::hull(&stats.close, 10);
    let _ = smooth::dema(&stats.close, 10);
}
