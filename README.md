# traquer

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

Indicators often behave quite similarly depending on window size and classfication/api may
change (if egregiously wrong).

### momentum
Provides technical indicators that measures the rate of change or speed of price
movement of a security. In the context of this library, these indicators are typically
range bound and/or centred around zero. These often begin to show trend the larger the
smoothing.

### trend
Indicators where the direction may signify opportunities. The slope and trajectory of the
indicator are more important than the magnitude of the resulting value.

### volatility
Indicators that measure the price movement, regardless of direction. In essence, it is
signaling whether there is a trend or not generally based on the delta between the
highest and lowest prices in period. It may also be represented as channels for which
it expects prices to fall within.

### volume
Indicators that factor in how much an asset has been traded in a period of time. Depending on
the indicator, it may be a momentum indicator or trend indicator.

### smooth
Provides moving average functions. Often used to track trend, levels of support, breakouts, etc...
The results are in the same scale as input data and is often used as a signal line for input data.


## TODO
- simplify reqs
- handle div by zero scenarios
- pad results with nan to match len of inputs
- allow other numeric types rather than just f64
