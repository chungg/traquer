//! Distribution Functions
//!
//! Provides functions which describe the distribution of a dataset. This can relate to the
//! shape, centre, or dispersion of the
//! distribution[[1]](https://en.wikipedia.org/wiki/Probability_distribution)
use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter;

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
pub fn variance(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            w.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2)) / window as f64
        }))
}

pub(crate) fn _std_dev(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.windows(window).map(move |w| {
        let mean = w.iter().sum::<f64>() / window as f64;
        (w.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2)) / window as f64).sqrt()
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
pub fn std_dev(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
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
pub fn zscore(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            let stdev =
                (w.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2)) / window as f64).sqrt();
            (w[window - 1] - mean) / stdev
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
pub fn mad(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            w.iter().fold(0.0, |acc, x| acc + (x - mean).abs()) / window as f64
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
pub fn cv(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            (w.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2)) / window as f64).sqrt() / mean
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
pub fn kurtosis(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let adj1 = ((window + 1) * window * (window - 1)) as f64 / ((window - 2) * (window - 3)) as f64;
    let adj2 = 3.0 * (window - 1).pow(2) as f64 / ((window - 2) * (window - 3)) as f64;
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            let k4 = w.iter().fold(0.0, |acc, x| acc + (x - mean).powi(4));
            let k2 = w.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2));
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
pub fn skew(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            let m3 = w.iter().fold(0.0, |acc, x| acc + (x - mean).powi(3)) / window as f64;
            let m2 = (w.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2)) / window as f64)
                .powf(3.0 / 2.0);
            ((window * (window - 1)) as f64).sqrt() / (window - 2) as f64 * m3 / m2
        }))
}

fn quickselect(data: &mut [f64], k: usize) -> f64 {
    // iterative solution is faster than recursive
    let mut lo = 0;
    let mut hi = data.len() - 1;
    while lo < hi {
        // Lomuto partition
        let pivot = data[hi];
        let mut i = lo;
        for j in lo..hi {
            if data[j] < pivot {
                data.swap(i, j);
                i += 1;
            }
        }
        data.swap(i, hi);

        match i.cmp(&k) {
            Ordering::Equal => return data[k],
            Ordering::Greater => hi = i - 1,
            Ordering::Less => lo = i + 1,
        };
    }
    data[k]
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
pub fn median(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mut w = w.to_vec();
            match window {
                even if even % 2 == 0 => {
                    let fst_med = quickselect(&mut w, (even / 2) - 1);
                    // don't qsort again, just get min of remainder
                    let snd_med = w[even / 2..]
                        .iter()
                        .fold(f64::NAN, |state, &x| state.min(x));

                    (fst_med + snd_med) * 0.5
                }
                odd => quickselect(&mut w, odd / 2),
            }
        }))
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
pub fn quantile(data: &[f64], window: usize, q: f64) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let pos = (window - 1) as f64 * (q.clamp(0.0, 100.0) / 100.0);
            let mut w = w.to_vec();
            match pos {
                exact if exact.fract() == 0.0 => quickselect(&mut w, exact as usize),
                _ => {
                    let lower = quickselect(&mut w, pos.floor() as usize);
                    // don't qsort again, just get min of remainder
                    let upper = w[pos.ceil() as usize..]
                        .iter()
                        .fold(f64::NAN, |state, &x| state.min(x));

                    lower * (pos.ceil() - pos) + upper * (pos - pos.floor())
                }
            }
        }))
}

pub(crate) fn cov_stdev<'a>(x: &'a [f64], y: &'a [f64]) -> (f64, f64, f64) {
    let x_avg = x.iter().fold(0.0, |acc, &x| acc + x) / x.len() as f64;
    let y_avg = y.iter().fold(0.0, |acc, &y| acc + y) / y.len() as f64;
    (
        x.iter()
            .zip(y)
            .fold(0.0, |acc, (&xi, &yi)| acc + (xi - x_avg) * (yi - y_avg))
            / (x.len() - 1) as f64,
        (x.iter().fold(0.0, |acc, &xi| acc + (xi - x_avg).powi(2)) / (x.len() - 1) as f64).sqrt(),
        (y.iter().fold(0.0, |acc, &yi| acc + (yi - y_avg).powi(2)) / (y.len() - 1) as f64).sqrt(),
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

// trait SortExt<T> {
//     // probably move this somewhere in future
//     fn argsort(&self) -> Vec<usize>;
// }
//
// impl<T: PartialOrd + Clone> SortExt<T> for Vec<T> {
//     fn argsort(&self) -> Vec<usize> {
//         let mut indices = (0..self.len()).collect::<Vec<_>>();
//         indices.sort_by(|&a, &b| self[a].partial_cmp(&self[b]).unwrap());
//         indices
//     }
// }

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
pub fn rank(data: &[f64], mode: Option<RankMode>) -> Box<dyn Iterator<Item = f64> + '_> {
    let mut values = data.to_vec();

    match mode {
        Some(RankMode::Ordinal) => {
            // argsort().argsort().iter().map(|&x| (x + 1) as f64) saves memory but is slower
            values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mut rank_map: HashMap<u64, (f64, f64)> = HashMap::new();
            values.iter().enumerate().for_each(|(rank, element)| {
                rank_map
                    .entry(element.to_bits())
                    .or_insert((rank as f64, 0.));
            });
            Box::new(data.iter().map(move |x| {
                let entry = rank_map.get(&x.to_bits()).unwrap().to_owned();
                rank_map.insert(x.to_bits(), (entry.0, entry.1 + 1.));
                entry.0 + entry.1 + 1.
            }))
        }
        _ => {
            values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mut rank_map: HashMap<u64, f64> = HashMap::new();

            match mode {
                Some(RankMode::Average) => {
                    let mut start_idx = 0;
                    for (idx, &element) in values.iter().enumerate().skip(1) {
                        if element != values[start_idx] {
                            rank_map.insert(
                                values[start_idx].to_bits(),
                                (start_idx..idx).sum::<usize>() as f64 / (idx - start_idx) as f64,
                            );
                            start_idx = idx;
                        }
                    }
                    rank_map.insert(
                        values[start_idx].to_bits(),
                        (start_idx..values.len()).sum::<usize>() as f64
                            / (values.len() - start_idx) as f64,
                    );
                }
                Some(RankMode::Max) => {
                    let size = values.len() - 1;
                    values.iter().rev().enumerate().for_each(|(rank, element)| {
                        rank_map
                            .entry(element.to_bits())
                            .or_insert((size - rank) as f64);
                    });
                }
                Some(RankMode::Dense) => {
                    let mut rank = 0.0;
                    values.iter().for_each(|element| {
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            rank_map.entry(element.to_bits())
                        {
                            e.insert(rank);
                            rank += 1.0;
                        }
                    });
                }
                _ => values.iter().enumerate().for_each(|(rank, element)| {
                    rank_map.entry(element.to_bits()).or_insert(rank as f64);
                }),
            };
            Box::new(
                data.iter()
                    .map(move |x| rank_map.get(&x.to_bits()).unwrap().to_owned() + 1.),
            )
        }
    }
}
