//! Correlation Indicators
//!
//! Provides technical indicators that measures how two or more series relate to one another.
//! These indicators can capture trend or performance and is often used to track relation
//! between an asset and a benchmark.
use std::iter;

use itertools::izip;

use crate::smooth;

/// Pearson Correlation Coefficient
///
/// The ratio between the covariance of two variables and the product of their standard
/// deviations; thus, it is essentially a normalized measurement of the covariance,
///
/// ```math
/// r_{xy} = \frac{n\sum x_i y_i - \sum x_i\sum y_i}{\sqrt{n\sum x_i^2-\left(\sum x_i\right)^2}~\sqrt{n\sum y_i^2-\left(\sum y_i\right)^2}}
/// ```
///
/// ## Usage
///
/// A value between 0 and 1 implies a positive correlation; 0, no correlation; and between 0 and
/// -1, a negative correlation.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
///
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::pcc(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn pcc<'a>(
    series1: &'a [f64],
    series2: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        series1
            .windows(window)
            .zip(series2.windows(window))
            .map(move |(x_win, y_win)| {
                let mut sum_xy = 0.0;
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_x2 = 0.0;
                let mut sum_y2 = 0.0;
                for (x, y) in x_win.iter().zip(y_win) {
                    sum_xy += x * y;
                    sum_x += x;
                    sum_y += y;
                    sum_x2 += x.powi(2);
                    sum_y2 += y.powi(2);
                }
                (window as f64 * sum_xy - sum_x * sum_y)
                    / ((window as f64 * sum_x2 - sum_x.powi(2)).sqrt()
                        * (window as f64 * sum_y2 - sum_y.powi(2)).sqrt())
            }),
    )
}

/// Coefficient of Determination (r-squared)
///
/// The square of the correlation coefficient. Examines how differences in one variable
/// can be explained by the difference in a second variable when predicting the outcome
/// of a given event.
///
/// ## Usage
///
/// A value of 1 indicates 100% correlation; 0, no correlation.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Coefficient_of_determination)
/// [[2]](https://danshiebler.com/2017-06-25-metrics/)
///
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::rsq(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn rsq<'a>(
    series1: &'a [f64],
    series2: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    pcc(series1, series2, window).map(|x| x.powi(2))
}

/// Beta Coefficient
///
/// Measure that compares the volatility of returns of an instrument to those of the market
/// as a whole.
///
/// ## Usage
///
/// Value of 1 suggests correlation with market. A value above 1 suggests the instrument is more
/// volatile compared to market and a value below 1 suggests, less volatility.
///
/// ## Sources
///
/// [[1]](https://seekingalpha.com/article/4493310-beta-for-stocks)
///
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::beta(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2).collect::<Vec<f64>>();
///
/// ```
pub fn beta<'a>(
    series1: &'a [f64],
    series2: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take((window - 1) * 2 + 1).chain(
        series1
            .windows(2)
            .map(|w| w[1] / w[0] - 1.0)
            .collect::<Vec<f64>>()
            .windows(window)
            .zip(
                series2
                    .windows(2)
                    .map(|w| w[1] / w[0] - 1.0)
                    .collect::<Vec<f64>>()
                    .windows(window),
            )
            .map(|(x_win, y_win)| {
                let sum_x = x_win.iter().sum::<f64>();
                let sum_y = y_win.iter().sum::<f64>();
                (
                    (x_win[window - 1] - sum_x / window as f64)
                        * (y_win[window - 1] - sum_y / window as f64),
                    (y_win[window - 1] - sum_y / window as f64).powi(2),
                )
            })
            .collect::<Vec<(f64, f64)>>()
            .windows(window)
            .map(|w| {
                let mut sum_covar = 0.0;
                let mut sum_var = 0.0;
                for (covar, var) in w {
                    sum_covar += covar;
                    sum_var += var;
                }
                (sum_covar / window as f64) / (sum_var / window as f64)
            })
            .collect::<Vec<f64>>(),
    )
}

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
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::perf(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2).collect::<Vec<f64>>();
///
/// ```
pub fn perf<'a>(
    series1: &'a [f64],
    series2: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    izip!(
        series1,
        series2,
        smooth::sma(series1, window),
        smooth::sma(series2, window)
    )
    .map(|(x, y, ma_x, ma_y)| x / y * (ma_y / ma_x))
}

/// Relative Strength Comparison (Price Relative)
///
/// Compares performance by computing a ratio simply dividing the base security by a benchmark.
///
/// ## Usage
///
/// An increasing value suggests base security is outperforming the benchmark.
///
/// ## Sources
///
/// [[1]](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/relative-strength-comparison)
/// [[2]](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/price-relative-relative-strength)
///
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::rsc(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     ).collect::<Vec<f64>>();
///
/// ```
pub fn rsc<'a>(series1: &'a [f64], series2: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    series1.iter().zip(series2).map(|(x, y)| x / y)
}
