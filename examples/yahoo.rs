use chrono::{NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};

use traquer::*;

#[derive(Deserialize, Serialize, Debug)]
struct SecStats {
    high: Vec<f64>,
    low: Vec<f64>,
    open: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<u64>,
}

fn get_prices(ticker: &str) -> SecStats {
    const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) \
    AppleWebKit/537.36 (KHTML, like Gecko) \
    Chrome/39.0.2171.95 Safari/537.36";

    let res = ureq::get(&format!(
        "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    ))
    .set("User-Agent", USER_AGENT)
    .query_pairs(vec![
        ("interval", "1d"),
        ("events", "capitalGain|div|split"),
        (
            "period1",
            &NaiveDateTime::parse_from_str("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
                .unwrap()
                .and_utc()
                .timestamp()
                .to_string(),
        ),
        ("period2", &Utc::now().timestamp().to_string()),
    ])
    .call()
    .unwrap();

    serde_json::from_value::<SecStats>(
        res.into_json::<serde_json::Value>().unwrap()["chart"]["result"][0]["indicators"]["quote"]
            [0]
        .clone(),
    )
    .unwrap()
}

fn main() {
    let stats = get_prices("SPY");
    dbg!(volume::kvo(
        &stats.high,
        &stats.low,
        &stats.close,
        &stats.volume,
        34,
        55,
        Some(false)
    )
    .collect::<Vec<_>>());
}
