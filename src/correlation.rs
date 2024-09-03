//! Correlation Indicators
//!
//! Provides technical indicators that measures how two or more series relate to one another.
//! These indicators can capture trend or performance and is often used to track relation
//! between an asset and a benchmark.
use std::iter;

use itertools::izip;
use num_traits::cast::ToPrimitive;

use crate::smooth;
use crate::statistic::distribution::{cov_stdev, rank, RankMode};

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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn pcc<'a, T: ToPrimitive>(
    series1: &'a [T],
    series2: &'a [T],
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
                    let (x, y) = (x.to_f64().unwrap(), y.to_f64().unwrap());
                    sum_xy += x * y;
                    sum_x += x;
                    sum_y += y;
                    sum_x2 += x * x;
                    sum_y2 += y * y;
                }
                (window as f64 * sum_xy - sum_x * sum_y)
                    / ((window as f64 * sum_x2 - sum_x * sum_x).sqrt()
                        * (window as f64 * sum_y2 - sum_y * sum_y).sqrt())
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn rsq<'a, T: ToPrimitive>(
    series1: &'a [T],
    series2: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    pcc(series1, series2, window).map(|x| x * x)
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2).collect::<Vec<f64>>();
///
/// ```
pub fn beta<'a, T: ToPrimitive>(
    series1: &'a [T],
    series2: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take((window - 1) * 2 + 1).chain(
        series1
            .windows(2)
            .map(|w| w[1].to_f64().unwrap() / w[0].to_f64().unwrap() - 1.0)
            .collect::<Vec<f64>>()
            .windows(window)
            .zip(
                series2
                    .windows(2)
                    .map(|w| w[1].to_f64().unwrap() / w[0].to_f64().unwrap() - 1.0)
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2).collect::<Vec<f64>>();
///
/// ```
pub fn perf<'a, T: ToPrimitive>(
    series1: &'a [T],
    series2: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    izip!(
        series1,
        series2,
        smooth::sma(series1, window),
        smooth::sma(series2, window)
    )
    .map(|(x, y, ma_x, ma_y)| x.to_f64().unwrap() / y.to_f64().unwrap() * (ma_y / ma_x))
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     ).collect::<Vec<f64>>();
///
/// ```
pub fn rsc<'a, T: ToPrimitive>(
    series1: &'a [T],
    series2: &'a [T],
) -> impl Iterator<Item = f64> + 'a {
    series1
        .iter()
        .zip(series2)
        .map(|(x, y)| x.to_f64().unwrap() / y.to_f64().unwrap())
}

/// Spearman's Rank Correlation Coefficient
///
/// Assesses how well the relationship between two variables can be described using a monotonic
/// function. Similar to Pearson correlation except uses rank value.
///
/// ## Usage
///
/// A value between 0 and 1 implies a positive correlation; 0, no correlation; and between 0 and
/// -1, a negative correlation.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
///
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::srcc(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn srcc<'a, T: ToPrimitive + PartialOrd + Clone>(
    series1: &'a [T],
    series2: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        series1
            .windows(window)
            .zip(series2.windows(window))
            .map(|(x_win, y_win)| {
                let (cov_xy, std_x, std_y) = cov_stdev(
                    &rank(x_win, None).collect::<Vec<f64>>(),
                    &rank(y_win, None).collect::<Vec<f64>>(),
                );
                cov_xy / (std_x * std_y)
            }),
    )
}

/// Kendall's Rank Correlation Coefficient
///
/// Meausres the similarity of the orderings of the data when ranked by each of the quantities by
/// computing the number of concordant and disconcordant pairs. This computes Tau-b matching
/// logic in scipy.
///
/// ## Usage
///
/// A value between 0 and 1 implies a positive correlation; 0, no correlation; and between 0 and
/// -1, a negative correlation.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)
/// [[2]](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)
///
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::krcc(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn krcc<'a, T: ToPrimitive>(
    series1: &'a [T],
    series2: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        series1
            .windows(window)
            .zip(series2.windows(window))
            .map(move |(x_win, y_win)| {
                let mut nc = 0.0;
                let mut x_tie = 0.0;
                let mut y_tie = 0.0;
                let mut xy_tie = 0.0;

                for i in 0..window - 1 {
                    for j in i + 1..window {
                        let (xi, xj, yi, yj) = (
                            x_win[i].to_f64().unwrap(),
                            x_win[j].to_f64().unwrap(),
                            y_win[i].to_f64().unwrap(),
                            y_win[j].to_f64().unwrap(),
                        );
                        nc += ((xi - xj).signum() == (yi - yj).signum() && xi != xj) as u8 as f64;
                        xy_tie += (xi == xj && yi == yj) as u8 as f64;
                        x_tie += (xi == xj) as u8 as f64;
                        y_tie += (yi == yj) as u8 as f64;
                    }
                }
                let tot = (window * (window - 1)) as f64 * 0.5;
                let nd = tot - nc - x_tie - y_tie + xy_tie;
                (nc - nd) / ((tot - x_tie) * (tot - y_tie)).sqrt()
            }),
    )
}

/// Hoeffding's D
///
/// A test based on the population measure of deviation from independence. More
/// resource-intensive compared to other correlation functions but may handle non-monotonic
/// relationships better.
///
/// ## Usage
///
/// Generates a measure that ranges from -0.5 to 1, where the higher the number is, the
/// more strongly dependent the two sequences are on each other.
///
/// ## Sources
///
/// [[1]](https://github.com/Dicklesworthstone/hoeffdings_d_explainer)
///
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::hoeffd(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn hoeffd<'a, T: ToPrimitive + PartialOrd + Clone>(
    series1: &'a [T],
    series2: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        series1
            .windows(window)
            .zip(series2.windows(window))
            .map(move |(x_win, y_win)| {
                let rank_x = rank(x_win, Some(RankMode::Average)).collect::<Vec<f64>>();
                let rank_y = rank(y_win, Some(RankMode::Average)).collect::<Vec<f64>>();
                let mut q = vec![1.0; window];
                for i in 0..window {
                    for j in 0..window {
                        q[i] += (rank_x[j] < rank_x[i] && rank_y[j] < rank_y[i]) as u8 as f64;
                        q[i] +=
                            0.25 * (rank_x[j] == rank_x[i] && rank_y[j] == rank_y[i]) as u8 as f64;
                        q[i] += 0.5
                            * ((rank_x[j] == rank_x[i] && rank_y[j] < rank_y[i])
                                || (rank_x[j] < rank_x[i] && rank_y[j] == rank_y[i]))
                                as u8 as f64;
                    }
                    q[i] -= 0.25; // accounts for when comparing to itself
                }
                let d1 = q.iter().fold(0.0, |acc, x| acc + (x - 1.0) * (x - 2.0));
                let d2 = rank_x.iter().zip(&rank_y).fold(0.0, |acc, (x, y)| {
                    acc + (x - 1.0) * (x - 2.0) * (y - 1.0) * (y - 2.0)
                });
                let d3 = izip!(q, rank_x, rank_y).fold(0.0, |acc, (q, x, y)| {
                    acc + (x - 2.0) * (y - 2.0) * (q - 1.0)
                });
                30.0 * (((window - 2) * (window - 3)) as f64 * d1 + d2
                    - 2. * (window - 2) as f64 * d3)
                    / (window * (window - 1) * (window - 2) * (window - 3) * (window - 4)) as f64
            }),
    )
}

/// Distance Correlation
///
/// Measures both linear and nonlinear association between two random variables or random vectors.
/// Not to be confused with correlation distance which is related to Pearson Coefficient[3].
///
/// ## Usage
///
/// Generates a value between 0 and 1 where 0 implies series are independent and 1 implies they are
/// surely equal.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Distance_correlation)
/// [[2]](https://www.freecodecamp.org/news/how-machines-make-predictions-finding-correlations-in-complex-data-dfd9f0d87889/)
/// [[3]](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Pearson's_distance)
///
/// # Examples
///
/// ```
/// use traquer::correlation;
///
/// correlation::dcor(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn dcor<'a, T: ToPrimitive>(
    series1: &'a [T],
    series2: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    fn centred_matrix<T: ToPrimitive>(x: &[T]) -> Vec<f64> {
        let n = x.len();
        // flattened NxN distance matrix, where [x_00..x0j, ... ,x_i0..x_ij]
        let mut matrix = vec![0.0; n * n];
        let mut matrix_sum = 0_f64;
        for i in 0..n {
            for j in i..n {
                let idx = (i * n) + j;
                let mirror_idx = idx / n + (idx % n) * n;
                matrix[idx] = (x[i].to_f64().unwrap() - x[j].to_f64().unwrap()).abs();
                matrix[mirror_idx] = matrix[idx];
                matrix_sum += matrix[idx] * 2.;
            }
        }

        // "double-centre" the matrix
        let row_means: Vec<f64> = (0..matrix.len())
            .step_by(n)
            .map(|i| matrix[i..i + n].iter().sum::<f64>() / n as f64)
            .collect();
        let col_means = &row_means;
        // undo the double count of mirror line rather than add if clause above
        matrix_sum -= (0..matrix.len())
            .step_by(n + 1)
            .fold(0.0, |acc, x| acc + matrix[x]);
        let matrix_mean: f64 = matrix_sum / (n * n) as f64;
        for (i, row_mean) in row_means.iter().enumerate() {
            for (j, col_mean) in col_means.iter().enumerate().skip(i) {
                let idx = (i * n) + j;
                let mirror_idx = idx / n + (idx % n) * n;
                matrix[idx] += -row_mean - col_mean + matrix_mean;
                matrix[mirror_idx] = matrix[idx];
            }
        }
        matrix
    }

    iter::repeat(f64::NAN).take(window - 1).chain(
        series1
            .windows(window)
            .zip(series2.windows(window))
            .map(move |(x_win, y_win)| {
                let centred_x = centred_matrix(x_win);
                let centred_y = centred_matrix(y_win);
                let dcov = (centred_x
                    .iter()
                    .zip(&centred_y)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / window.pow(2) as f64)
                    .sqrt();
                let dvar_x =
                    (centred_x.iter().map(|a| a * a).sum::<f64>() / window.pow(2) as f64).sqrt();
                let dvar_y =
                    (centred_y.iter().map(|a| a * a).sum::<f64>() / window.pow(2) as f64).sqrt();

                dcov / (dvar_x * dvar_y).sqrt()
            }),
    )
}
