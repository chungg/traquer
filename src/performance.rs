use itertools::izip;

use crate::smooth;

/// Performance Index
///
/// Used to compare a asset's price trend to the general trend of a benchmark index.
///
/// ## Usage
///
/// Value above 1 would state that base series outperforms benchmark.
///
/// ## Sources
///
/// [[1]](https://www.marketvolume.com/technicalanalysis/performanceindex.asp)
///
///
pub fn perf_idx<'a>(
    base: &'a [f64],
    bench: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    izip!(
        base,
        bench,
        smooth::sma(base, window),
        smooth::sma(bench, window)
    )
    .map(|(x, y, ma_x, ma_y)| x / y * (ma_y / ma_x))
}

/// Drawdown
///
/// Measures an investment or trading account's decline from the peak before it recovers back
/// to that peak.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/d/drawdown.asp)
pub fn drawdown(data: &[f64]) -> impl Iterator<Item = f64> + '_ {
    data.iter().scan(data[0], |state, &x| {
        *state = state.max(x);
        Some(1.0 - (x / *state))
    })
}
