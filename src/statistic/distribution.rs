//! Distribution Functions
//!
//! Provides functions which describe the distribution of a dataset. This can relate to the
//! shape, centre, or dispersion of the
//! distribution[[1]](https://en.wikipedia.org/wiki/Probability_distribution)
use std::cmp::Ordering;
use std::iter;

use num_traits::cast::ToPrimitive;

/// Variance
///
/// A measure of how far a set of numbers is spread out from their average value. Provides
/// biased population variance.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Variance)
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::variance(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn variance<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().filter_map(|x| x.to_f64()).sum::<f64>() / window as f64;
            w.iter()
                .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).powi(2))
                / window as f64
        }))
}

pub(crate) fn _std_dev<T: ToPrimitive>(
    data: &[T],
    window: usize,
) -> impl Iterator<Item = f64> + '_ {
    data.windows(window).map(move |w| {
        let mean = w.iter().filter_map(|x| x.to_f64()).sum::<f64>() / window as f64;
        (w.iter()
            .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).powi(2))
            / window as f64)
            .sqrt()
    })
}

/// Standard Deviation
///
/// A measure of the amount of variation of a random variable expected about its mean.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Standard_deviation)
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::std_dev(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn std_dev<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(_std_dev(data, window))
}

/// Standard Score (Z-score)
///
/// Computes the number of standard deviations a value is above or below its mean.
///
/// ```math
/// z = {x- \mu \over \sigma}
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Standard_score)
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::zscore(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn zscore<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().filter_map(|x| x.to_f64()).sum::<f64>() / window as f64;
            let stdev = (w
                .iter()
                .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).powi(2))
                / window as f64)
                .sqrt();
            (w[window - 1].to_f64().unwrap() - mean) / stdev
        }))
}

/// Mean Absolute Deviation
///
/// A measure of variability that indicates the average distance between observations and
/// their mean. An alternative to Standard Deviation.
///
/// ```math
/// \frac{1}{n} \sum_{i=1}^n |x_i-m(X)|
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Average_absolute_deviation)
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::mad(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn mad<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().filter_map(|x| x.to_f64()).sum::<f64>() / window as f64;
            w.iter()
                .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).abs())
                / window as f64
        }))
}

/// Coefficient of Variation
///
/// Measure of dispersion of a probability distribution or frequency distribution. Another
/// alternative to standard deviation. The actual value of the CV is independent of the unit
/// in which the measurement has been taken which can be advantageous.
///
/// ```math
/// CV = \frac{\sigma}{\mu}
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Coefficient_of_variation)
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::cv(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn cv<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().filter_map(|x| x.to_f64()).sum::<f64>() / window as f64;
            (w.iter()
                .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).powi(2))
                / window as f64)
                .sqrt()
                / mean
        }))
}

/// Kurtosis
///
/// A measure of the "tailedness" of the probability distribution of a real-valued random
/// variable. Computes the standard unbiased estimator.
///
/// ```math
/// \frac{(n+1)\,n\,(n-1)}{(n-2)\,(n-3)} \; \frac{\sum_{i=1}^n (x_i - \bar{x})^4}{\left(\sum_{i=1}^n (x_i - \bar{x})^2\right)^2} - 3\,\frac{(n-1)^2}{(n-2)\,(n-3)} \\[6pt]
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Kurtosis)
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::kurtosis(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn kurtosis<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let adj1 = ((window + 1) * window * (window - 1)) as f64 / ((window - 2) * (window - 3)) as f64;
    let adj2 = 3.0 * (window - 1).pow(2) as f64 / ((window - 2) * (window - 3)) as f64;
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().filter_map(|x| x.to_f64()).sum::<f64>() / window as f64;
            let k4 = w
                .iter()
                .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).powi(4));
            let k2 = w
                .iter()
                .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).powi(2));
            adj1 * k4 / k2.powi(2) - adj2
        }))
}

/// Skew
///
/// The degree of asymmetry observed in a probability distribution. When data points on a
/// bell curve are not distributed symmetrically to the left and right sides of the median,
/// the bell curve is skewed. Distributions can be positive and right-skewed, or negative
/// and left-skewed.
///
/// Computes unbiased adjusted Fisher–Pearson standardized moment coefficient G1
///
/// ```math
/// g_1 = \frac{m_3}{m_2^{3/2}}
///     = \frac{\tfrac{1}{n} \sum_{i=1}^n (x_i-\overline{x})^3}{\left[\tfrac{1}{n} \sum_{i=1}^n (x_i-\overline{x})^2 \right]^{3/2}},
///
/// G_1 = \frac{k_3}{k_2^{3/2}} = \frac{n^2}{(n-1)(n-2)}\; b_1 = \frac{\sqrt{n(n-1)}}{n-2}\; g_1
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Skewness)
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::skew(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn skew<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().filter_map(|x| x.to_f64()).sum::<f64>() / window as f64;
            let m3 = w
                .iter()
                .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).powi(3))
                / window as f64;
            let m2 = (w
                .iter()
                .fold(0.0, |acc, x| acc + (x.to_f64().unwrap() - mean).powi(2))
                / window as f64)
                .powf(3.0 / 2.0);
            ((window * (window - 1)) as f64).sqrt() / (window - 2) as f64 * m3 / m2
        }))
}

fn quickselect<T: ToPrimitive + PartialOrd + Clone>(data: &mut [T], k: usize) -> T {
    // iterative solution is faster than recursive
    let mut lo = 0;
    let mut hi = data.len() - 1;
    while lo < hi {
        // Lomuto partition
        let pivot = data[hi].clone();
        let mut i = lo;
        for j in lo..hi {
            if data[j] < pivot {
                data.swap(i, j);
                i += 1;
            }
        }
        data.swap(i, hi);

        match i.cmp(&k) {
            Ordering::Equal => return data[k].clone(),
            Ordering::Greater => hi = i - 1,
            Ordering::Less => lo = i + 1,
        };
    }
    data[k].clone()
}

/// Median
///
/// The value separating the higher half from the lower half of a data sample, a population, or a
/// probability distribution within a window.
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::median(&vec![1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn median<T: ToPrimitive + PartialOrd + Clone>(
    data: &[T],
    window: usize,
) -> impl Iterator<Item = f64> + '_ {
    quantile(data, window, 50.0)
}

/// Quantile
///
/// Partition a finite set of values into q subsets of (nearly) equal sizes. 50th percentile is
/// equivalent to median.
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::quantile(&vec![1.0,2.0,3.0,4.0,5.0], 3, 90.0).collect::<Vec<f64>>();
/// ```
pub fn quantile<T: ToPrimitive + PartialOrd + Clone>(
    data: &[T],
    window: usize,
    q: f64,
) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let pos = (window - 1) as f64 * (q.clamp(0.0, 100.0) / 100.0);
            let mut w = w.to_vec();
            match pos {
                exact if exact.fract() == 0.0 => {
                    quickselect(&mut w, exact as usize).to_f64().unwrap()
                }
                _ => {
                    let lower = quickselect(&mut w, pos.floor() as usize).to_f64().unwrap();
                    // don't qsort again, just get min of remainder
                    let upper = w[pos.ceil() as usize..]
                        .iter()
                        .fold(f64::NAN, |state, x| state.min(x.to_f64().unwrap()));

                    lower * (pos.ceil() - pos) + upper * (pos - pos.floor())
                }
            }
        }))
}

pub(crate) fn cov_stdev<'a, T: ToPrimitive>(x: &'a [T], y: &'a [T]) -> (f64, f64, f64) {
    let x_avg = x.iter().fold(0.0, |acc, x| acc + x.to_f64().unwrap()) / x.len() as f64;
    let y_avg = y.iter().fold(0.0, |acc, y| acc + y.to_f64().unwrap()) / y.len() as f64;
    (
        x.iter().zip(y).fold(0.0, |acc, (xi, yi)| {
            acc + (xi.to_f64().unwrap() - x_avg) * (yi.to_f64().unwrap() - y_avg)
        }) / (x.len() - 1) as f64,
        (x.iter()
            .fold(0.0, |acc, xi| acc + (xi.to_f64().unwrap() - x_avg).powi(2))
            / (x.len() - 1) as f64)
            .sqrt(),
        (y.iter()
            .fold(0.0, |acc, yi| acc + (yi.to_f64().unwrap() - y_avg).powi(2))
            / (y.len() - 1) as f64)
            .sqrt(),
    )
}

/// Rank modes types
pub enum RankMode {
    Average,
    Dense,
    Max,
    Min,
    Ordinal,
}

trait SortExt<T> {
    // probably move this somewhere in future
    fn argsort(&self) -> Vec<usize>;
}

impl<T: PartialOrd + Clone> SortExt<T> for [T] {
    fn argsort(&self) -> Vec<usize> {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&a, &b| self[a].partial_cmp(&self[b]).unwrap());
        indices
    }
}

/// Rank
///
/// Assign ranks to data, dealing with ties appropriately. Ranks begin at 1.
///
/// Different ranking options are available:
///
/// - `Average`: The average of the ranks that would have been assigned to all the tied values is
///   assigned to each value.
/// - `Min`: The minimum of the ranks that would have been assigned to all the tied values is
///   assigned to each value. (This is also referred to as “competition” ranking.)
/// - `Max`: The maximum of the ranks that would have been assigned to all the tied values is
///   assigned to each value.
/// - `Dense`: Like `Min`, but the rank of the next highest element is assigned the rank
///   immediately after those assigned to the tied elements.
/// - `Ordinal`: All values are given a distinct rank, corresponding to the order that the values
///   occur in input.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Ranking_(statistics))
///
/// # Examples
///
/// ```
/// use traquer::statistic::distribution;
///
/// distribution::rank(
///     &[1.0,2.0,3.0,4.0,5.0],
///     Some(distribution::RankMode::Ordinal)
/// ).collect::<Vec<_>>();
/// ```
pub fn rank<T: ToPrimitive + PartialOrd + Clone>(
    data: &[T],
    mode: Option<RankMode>,
) -> impl Iterator<Item = f64> + '_ {
    let mut result = vec![0.; data.len()];
    let indices = data.argsort();

    match mode {
        Some(RankMode::Ordinal) => {
            (1..=data.len())
                .zip(indices)
                .for_each(|(i, val)| result[val] = i as f64);
        }
        _ => {
            let mut sorted = indices.iter().map(|&i| data[i].clone());
            let x1 = sorted.next().unwrap();
            let uniq_indices = iter::once(true)
                .chain(sorted.scan(x1, |state, x| {
                    let result = *state != x;
                    *state = x;
                    Some(result)
                }))
                .collect::<Vec<bool>>();
            let ranks: Box<dyn Iterator<Item = f64>> = match mode {
                Some(RankMode::Dense) => Box::new(uniq_indices.iter().scan(0, |state, &take| {
                    if take {
                        *state += 1;
                    }
                    Some(*state as f64)
                })),
                Some(RankMode::Average) | Some(RankMode::Max) => {
                    let counts = uniq_indices
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &take)| if take { Some(i) } else { None })
                        .chain(iter::once(data.len()))
                        .collect::<Vec<usize>>();
                    match mode {
                        Some(RankMode::Average) => {
                            Box::new(uniq_indices.iter().scan(0, move |state, &take| {
                                if take {
                                    *state += 1;
                                }
                                Some((1 + counts[*state] + counts[*state - 1]) as f64 / 2.)
                            }))
                        }
                        _ => Box::new(uniq_indices.iter().scan(0, move |state, &take| {
                            if take {
                                *state += 1;
                            }
                            Some(counts[*state] as f64)
                        })),
                    }
                }
                _ => Box::new(
                    (1..=data.len())
                        .zip(uniq_indices)
                        .scan(1, |state, (rank, take)| {
                            if take {
                                *state = rank;
                                Some(rank as f64)
                            } else {
                                Some(*state as f64)
                            }
                        }),
                ),
            };
            ranks.zip(indices).for_each(|(i, val)| result[val] = i);
        }
    }
    result.into_iter()
}
