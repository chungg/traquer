[![Crates.io][crates-badge]][crates-url]
[![Apache licensed][license-badge]][license-url]
[![Build Status][actions-badge]][actions-url]

[crates-badge]: https://img.shields.io/crates/v/traquer.svg
[crates-url]: https://crates.io/crates/traquer
[license-badge]: https://img.shields.io/badge/license-Apache%20v2-blue.svg
[license-url]: https://github.com/chungg/traquer/blob/main/LICENSE
[actions-badge]: https://github.com/chungg/traquer/actions/workflows/gate.yml/badge.svg
[actions-url]: https://github.com/chungg/traquer/actions?query=branch%3Amain

# traquer

A simple, dataframe-agnostic, technical analysis library that gives you false hope
that you can beat the market.

## types of indicators

130+ indicators available across multiple categories. Even across categories, indicators often
behave quite similarly depending on window size. The classfication/api may change
(if egregiously wrong).

### momentum
Provides technical indicators that measures the rate of change or speed of price
movement of a security. In the context of this library, these indicators are typically
range bound and/or centred around zero. These often implicitly show trend.

### trend
Indicators where the direction may signify opportunities. The slope and trajectory of the
indicator are more important than the value alone.

### volatility
Indicators that measure the price movement, regardless of direction. In essence, it is
signaling whether there is a trend or not, generally based on the delta between the
highest and lowest prices in period. It may also be represented as channels for which
it expects prices to fall within.

### volume
Indicators that factor in how much an asset has been traded in a period of time. Depending on
the indicator, it may be a momentum indicator or trend indicator.

### smooth
Provides moving average functions. Often used to track trend, levels of support, breakouts, etc...
The results are in the same scale as input data and are often used as a signal line for input data.

### correlation
Signals that compare two or more variables and their relationship to one another.

### statistic
A set of common statistical functions that can describe features of a dataset or infer
conclusions such as prediction accuracy or patterns.

## installation
1. (optional) https://rustup.rs/
2. (optional) cargo new <lib name>
3. cargo add traquer

## quick start

```rust
use traquer::smooth;

smooth::ewma(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
```

## contributing
encouraged.

- git clone git@github.com:chungg/traquer.git
- https://rustup.rs/
- cargo test
- cargo bench
- cargo run --example file_json

## todo
- handle div by zero scenarios
