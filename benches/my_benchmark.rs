//use criterion::{black_box, criterion_group, criterion_main, Criterion};
//
//use signals::*;
//
//pub fn criterion_benchmark(c: &mut Criterion) {
//    c.bench_function("fib 20", |b| b.iter(|| black_box(main())));
//}
//
//criterion_group!(benches, criterion_benchmark);
//criterion_main!(benches);

#[divan::bench]
fn run() {
    divan::black_box(signals::main());
}

fn main() {
    divan::main();
}
