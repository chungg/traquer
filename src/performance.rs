use itertools::izip;

use crate::smooth;

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
