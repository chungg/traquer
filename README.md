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
### momentum
Indicators where crossing a threshold (zero line, signal line, etc...) may signify opportunities.
These thresholds may be: a signal line, zero lines, upper/lower value bounds, etc...

### trend
Indicators where the direction may signify opportunities. The slope and trajectory of the
indicator are more important than the actual value.

### volume
Indicators that factor in how much an asset has been traded in a period of time. Depending on
indicator, it may be a momentum indicator or trend indicator.

### smooth
Provides moving average functions. Often used to track trend, levels of support, breakouts, etc...
The results are in the same scale as input data and is often used as a signal line for input data.


## TODO
- classify indicators better
- simplify reqs
- handle div by zero scenarios
- pad results with nan to match len of inputs
- allow other numeric types rather than just f64
