# traquer

WIP - api is not stable.

Technical analysis library that gives you false hope that you can beat the market.

## installation
1. (optional) https://rustup.rs/
2. (optional) cargo new <lib name>
3. cargo add traquer

## quick start

```rust
use traquer::smooth;

fn main() {
    dbg!(smooth::ewma(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>());
}
```

## contributing
encouraged.

- git clone git@github.com:chungg/traquer.git
- https://rustup.rs/
- cargo test
- cargo bench

## types of indicators
- momentum - indicators where crossing a threshold (zero line) may signify opportunities.
- trend - indicators where the direction (uptrend/downtrend) may signify opportunities.
- volume - indicators that factor in how much an asset has been traded in a period of time.
- smooth - moving average functions. often used to track trend, levels of support, breakouts,
           etc... Is the same scale as input data.


## TODO
- docs
- simplify reqs
- more algos
- handle div by zero scenarios
- return iterable instead vec
- pad results with nan to match len of inputs
- allow other numeric types rather than just f64
