use criterion::{black_box, criterion_group, criterion_main, Criterion};

use serde::{Deserialize, Serialize};
use std::fs;

use traquer::indicator;

#[derive(Deserialize, Serialize, Debug)]
struct SecStats {
    high: Vec<f64>,
    low: Vec<f64>,
    open: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = fs::read_to_string("./aapl.input").expect("Unable to read file");
    let stats: SecStats = serde_json::from_str(&data).expect("JSON does not have correct format.");
    c.bench_function("fib 20", |b| {
        b.iter(|| {
            black_box(indicator::ultimate(
                &stats.high,
                &stats.low,
                &stats.close,
                6,
                12,
                24,
            ))
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

//#[divan::bench]
//fn run() {
//    divan::black_box(traquer::main());
//}
//
//fn main() {
//    divan::main();
//}
