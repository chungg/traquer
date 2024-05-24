use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn criterion_benchmark(c: &mut Criterion) {
    let data = fs::read_to_string("./aapl.input").expect("Unable to read file");
    let stats: SecStats = serde_json::from_str(&data).expect("JSON does not have correct format.");
    c.bench_function("sig-adx", |b| {
        b.iter(|| {
            black_box(indicator::adx(
                &stats.high,
                &stats.low,
                &stats.close,
                14,
                14,
            ))
        })
    });
    c.bench_function("sig-qstick", |b| {
        b.iter(|| black_box(indicator::qstick(&stats.open, &stats.close, 8)))
    });
    c.bench_function("sig-twiggs", |b| {
        b.iter(|| {
            black_box(volume::twiggs(
                &stats.high,
                &stats.low,
                &stats.close,
                &stats.volume,
                16,
            ))
        })
    });
    c.bench_function("sig-rsi", |b| {
        b.iter(|| black_box(indicator::rsi(&stats.close, 16)))
    });
    c.bench_function("sig-kvo", |b| {
        b.iter(|| {
            black_box(volume::kvo(
                &stats.high,
                &stats.low,
                &stats.close,
                &stats.volume,
                10,
                16,
            ))
        })
    });
    c.bench_function("sig-macd", |b| {
        b.iter(|| black_box(indicator::macd(&stats.close, 12, 26)))
    });
    c.bench_function("sig-cmo", |b| {
        b.iter(|| black_box(indicator::cmo(&stats.close, 16)))
    });
    c.bench_function("sig-cog", |b| {
        b.iter(|| black_box(indicator::cog(&stats.close, 16)))
    });
    c.bench_function("sig-shinohara", |b| {
        b.iter(|| {
            black_box(indicator::shinohara(
                &stats.high,
                &stats.low,
                &stats.close,
                26,
            ))
        })
    });
    c.bench_function("sig-elder_ray", |b| {
        b.iter(|| {
            black_box(indicator::elder_ray(
                &stats.high,
                &stats.low,
                &stats.close,
                16,
            ))
        })
    });
    c.bench_function("sig-elder_force", |b| {
        b.iter(|| black_box(volume::elder_force(&stats.close, &stats.volume, 16)))
    });
    c.bench_function("sig-mfi", |b| {
        b.iter(|| {
            black_box(volume::mfi(
                &stats.high,
                &stats.low,
                &stats.close,
                &stats.volume,
                16,
            ))
        })
    });
    c.bench_function("sig-ad", |b| {
        b.iter(|| {
            black_box(volume::ad(
                &stats.high,
                &stats.low,
                &stats.close,
                &stats.volume,
            ))
        })
    });
    c.bench_function("sig-ad_yahoo", |b| {
        b.iter(|| {
            black_box(volume::ad_yahoo(
                &stats.high,
                &stats.low,
                &stats.close,
                &stats.volume,
            ))
        })
    });
    c.bench_function("sig-cmf", |b| {
        b.iter(|| {
            black_box(volume::cmf(
                &stats.high,
                &stats.low,
                &stats.close,
                &stats.volume,
                16,
            ))
        })
    });
    c.bench_function("sig-cvi", |b| {
        b.iter(|| black_box(indicator::cvi(&stats.high, &stats.low, 16, 2)))
    });
    c.bench_function("sig-wpr", |b| {
        b.iter(|| black_box(indicator::wpr(&stats.high, &stats.low, &stats.close, 16)))
    });
    c.bench_function("sig-vortex", |b| {
        b.iter(|| black_box(indicator::vortex(&stats.high, &stats.low, &stats.close, 16)))
    });
    c.bench_function("sig-ppo", |b| {
        b.iter(|| black_box(indicator::ppo(&stats.volume, 10, 16)))
    });
    c.bench_function("sig-apo", |b| {
        b.iter(|| black_box(indicator::apo(&stats.close, 10, 16)))
    });
    c.bench_function("sig-dpo", |b| {
        b.iter(|| black_box(indicator::dpo(&stats.close, 16)))
    });
    c.bench_function("sig-vhf", |b| {
        b.iter(|| black_box(indicator::vhf(&stats.high, &stats.low, &stats.close, 16)))
    });
    c.bench_function("sig-ultimate", |b| {
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
    c.bench_function("sig-pgo", |b| {
        b.iter(|| black_box(indicator::pgo(&stats.high, &stats.low, &stats.close, 16)))
    });
    c.bench_function("sig-si", |b| {
        b.iter(|| {
            black_box(indicator::si(
                &stats.open,
                &stats.high,
                &stats.low,
                &stats.close,
                0.5,
            ))
        })
    });
    c.bench_function("sig-asi", |b| {
        b.iter(|| {
            black_box(indicator::asi(
                &stats.open,
                &stats.high,
                &stats.low,
                &stats.close,
                0.5,
            ))
        })
    });
    c.bench_function("sig-ulcer", |b| {
        b.iter(|| black_box(indicator::ulcer(&stats.close, 8)))
    });
    c.bench_function("sig-tr", |b| {
        b.iter(|| black_box(indicator::tr(&stats.high, &stats.low, &stats.close)))
    });
    c.bench_function("sig-hlc3", |b| {
        b.iter(|| black_box(indicator::hlc3(&stats.high, &stats.low, &stats.close, 16)))
    });
    c.bench_function("sig-trix", |b| {
        b.iter(|| black_box(indicator::trix(&stats.close, 7)))
    });
    c.bench_function("sig-tii", |b| {
        b.iter(|| black_box(indicator::tii(&stats.close, 16)))
    });
    c.bench_function("sig-tvi", |b| {
        b.iter(|| black_box(volume::tvi(&stats.close, &stats.volume, 0.5)))
    });
    c.bench_function("sig-supertrend", |b| {
        b.iter(|| {
            black_box(indicator::supertrend(
                &stats.high,
                &stats.low,
                &stats.close,
                16,
                3.0,
            ))
        })
    });
    c.bench_function("sig-stochastic", |b| {
        b.iter(|| {
            black_box(indicator::stochastic(
                &stats.high,
                &stats.low,
                &stats.close,
                16,
            ))
        })
    });
    c.bench_function("sig-stc", |b| {
        b.iter(|| black_box(indicator::stc(&stats.close, 3, 6, 12)))
    });
    c.bench_function("sig-relative_vol", |b| {
        b.iter(|| black_box(indicator::relative_vol(&stats.close, 6, 10)))
    });
    c.bench_function("sig-relative_vigor", |b| {
        b.iter(|| {
            black_box(indicator::relative_vigor(
                &stats.open,
                &stats.high,
                &stats.low,
                &stats.close,
                16,
            ))
        })
    });
    c.bench_function("sig-rwi", |b| {
        b.iter(|| black_box(indicator::rwi(&stats.high, &stats.low, &stats.close, 16)))
    });
    c.bench_function("sig-fisher", |b| {
        b.iter(|| black_box(indicator::fisher(&stats.high, &stats.low, 16)))
    });
    c.bench_function("sig-rainbow", |b| {
        b.iter(|| black_box(indicator::rainbow(&stats.close, 3, 16)))
    });
    c.bench_function("sig-ease", |b| {
        b.iter(|| black_box(volume::ease(&stats.high, &stats.low, &stats.volume, 16)))
    });
    c.bench_function("sig-obv", |b| {
        b.iter(|| black_box(volume::obv(&stats.close, &stats.volume)))
    });
    c.bench_function("ma-ewma", |b| {
        b.iter(|| black_box(smooth::ewma(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-sma", |b| {
        b.iter(|| black_box(smooth::sma(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-dema", |b| {
        b.iter(|| black_box(smooth::dema(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-tema", |b| {
        b.iter(|| black_box(smooth::tema(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-wma", |b| {
        b.iter(|| black_box(smooth::wma(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-wilder", |b| {
        b.iter(|| black_box(smooth::wilder(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-hull", |b| {
        b.iter(|| black_box(smooth::hull(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-vidya", |b| {
        b.iter(|| black_box(smooth::vidya(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-vma", |b| {
        b.iter(|| black_box(smooth::vma(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-lrf", |b| {
        b.iter(|| black_box(smooth::lrf(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-trima", |b| {
        b.iter(|| black_box(smooth::trima(&stats.close, 16).collect::<Vec<f64>>()))
    });
    c.bench_function("ma-zlma", |b| {
        b.iter(|| black_box(smooth::zlma(&stats.close, 16).collect::<Vec<f64>>()))
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
