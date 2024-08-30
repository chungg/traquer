//! Moving Average Functions
//!
//! Provides moving average functions. Often used to track trend, levels of support,
//! breakouts, etc... The results are in the same scale as input data and are often used
//! as a signal line for input data.

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::iter;

use itertools::izip;
use num_traits::cast::ToPrimitive;

use crate::statistic::distribution::_std_dev;

/// Moving average types
pub enum MaMode {
    SMA,
    EWMA,
    DEMA,
    TEMA,
    WMA,
    Hull,
    LinReg,
    Pascal,
    Triangle,
    Variable,
    Vidya,
    Wilder,
    ZeroLag,
}

/// Generic function to "dynamically" select moving average algorithm
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::ma(&[1.0,2.0,3.0,4.0,5.0], 3, smooth::MaMode::SMA).collect::<Vec<f64>>();
/// ```
pub fn ma<T: ToPrimitive>(
    data: &[T],
    window: usize,
    mamode: MaMode,
) -> Box<dyn Iterator<Item = f64> + '_> {
    match mamode {
        MaMode::SMA => Box::new(sma(data, window)),
        MaMode::EWMA => Box::new(ewma(data, window)),
        MaMode::DEMA => Box::new(dema(data, window)),
        MaMode::TEMA => Box::new(tema(data, window)),
        MaMode::WMA => Box::new(wma(data, window)),
        MaMode::Hull => Box::new(hull(data, window)),
        MaMode::LinReg => Box::new(lrf(data, window)),
        MaMode::Pascal => Box::new(pwma(data, window)),
        MaMode::Triangle => Box::new(trima(data, window)),
        MaMode::Variable => Box::new(vma(data, window)),
        MaMode::Vidya => Box::new(vidya(data, window)),
        MaMode::Wilder => Box::new(wilder(data, window)),
        MaMode::ZeroLag => Box::new(zlma(data, window)),
    }
}

/// Exponentially Weighted Moving Average
///
/// A type of moving average that gives more weight to more recent data points,
/// with the weight decreasing exponentially as you move further back in time.
///
/// EWMA = (α * newest_data_point) + ((1 - α) * previous_EWMA)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::ewma(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn ewma<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let initial = data[..window]
        .iter()
        .filter_map(|x| x.to_f64())
        .sum::<f64>()
        / window as f64;
    let alpha = 2.0 / (window + 1) as f64;
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(
            iter::once(initial).chain(data[window..].iter().scan(initial, move |state, x| {
                *state = x.to_f64().unwrap() * alpha + *state * (1.0 - alpha);
                Some(*state)
            })),
        )
}

/// Simple Moving Average
///
/// A type of moving average that gives equal weight to all data points in a fixed window.
///
/// SMA = (sum of n previous data points) / n
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::sma(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn sma<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(window - 1).chain(
        data.windows(window)
            .map(move |w| w.iter().filter_map(|x| x.to_f64()).sum::<f64>() / window as f64),
    )
}

/// Double Exponential Moving Average
///
/// A type of moving average that gives more weight to recent data points and adapts
/// quickly to changes in the trend.
///
/// DEMA = 2 * EMA1 - EMA2
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/d/double-exponential-moving-average.asp)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::dema(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn dema<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let ma = ewma(data, window).collect::<Vec<f64>>();
    let mama = iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(ewma(&ma[window - 1..], window).collect::<Vec<f64>>());
    ma.into_iter().zip(mama).map(|(ma1, ma2)| 2.0 * ma1 - ma2)
}

/// Triple Exponential Moving Average
///
/// A type of moving average that gives more weight to recent data points and adapts
/// quickly to changes in the trend.
///
/// TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::tema(&[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn tema<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let ma = ewma(data, window).collect::<Vec<f64>>();
    let ma2 = ewma(&ma[window - 1..], window).collect::<Vec<f64>>();
    let ma3 = ewma(&ma2[window - 1..], window).collect::<Vec<f64>>();
    izip!(
        ma,
        iter::repeat(f64::NAN).take(window - 1).chain(ma2),
        iter::repeat(f64::NAN).take((window - 1) * 2).chain(ma3),
    )
    .map(|(ma1, ma2, ma3)| 3.0 * ma1 - 3.0 * ma2 + ma3)
}

/// Weighted Moving Average
///
/// Assigns different weights to each data point in a moving average calculation unlike
/// a Simple Moving Average (SMA), which gives equal weight to all data points. Uses linear
/// weights.
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::wma(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn wma<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let denom = (window.pow(2) + window) as f64 / 2.0;
    let weights: Vec<f64> = (1..=window).map(|i| i as f64 / denom).collect();
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            w.iter()
                .zip(weights.iter())
                .map(|(value, weight)| value.to_f64().unwrap() * weight)
                .sum()
        }))
}

/// Pascal's Triangle Moving Average
///
/// Uses the coefficients from Pascal's Triangle to weight the data points in a moving average
/// calculation.
///
/// PWMA = (C(n,0) * x1 + C(n,1) * x2 + ... + C(n,n) * xn) / 2^n
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Pascal%27s_triangle)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::pwma(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn pwma<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let n = window - 1;
    let mut row = vec![1; window];
    let mut denom = 2_usize;
    for i in 1..=(n / 2) {
        let k = i - 1;
        row[i] = (row[k] * (n - k)) / (k + 1);
        row[n - i] = row[i];
        denom += row[i] * 2;
    }
    if window % 2 == 1 {
        denom -= row[n / 2];
    }
    let weights: Vec<f64> = row.into_iter().map(|i| i as f64 / denom as f64).collect();
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            w.iter()
                .zip(weights.iter())
                .map(|(value, weight)| value.to_f64().unwrap() * weight)
                .sum()
        }))
}

/// Welles Wilder's Moving Average (aka Smoothed MA aka Running MA)
///
/// Developed by J. Welles Wilder Jr. A type of moving average that uses a smoothing
/// formula to reduce the lag and volatility associated with traditional moving averages.
///
/// Wilder = (Sum of (n-1) previous prices + current price) / n
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::wilder(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn wilder<T: ToPrimitive>(data: &[T], window: usize) -> Box<dyn Iterator<Item = f64> + '_> {
    if window == 1 {
        return Box::new(data.iter().filter_map(|x| x.to_f64()));
    }
    let initial = data[..window - 1]
        .iter()
        .filter_map(|x| x.to_f64())
        .sum::<f64>()
        / (window - 1) as f64;
    Box::new(
        iter::repeat(f64::NAN)
            .take(window - 1)
            .chain(data[window - 1..].iter().scan(initial, move |state, x| {
                *state = (*state * (window - 1) as f64 + x.to_f64().unwrap()) / window as f64;
                Some(*state)
            })),
    )
}

/// Hull's Moving Average
///
/// Developed by Alan Hull. A type of moving average that uses a weighted average of three
/// different moving averages to create a more responsive and accurate indicator.
///
/// Hull = WMA(2*sqrt(n)) * 2 - WMA(sqrt(n))
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::hull(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn hull<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let ma = wma(data, window);
    let ma2 = wma(data, window.div_ceil(2));
    wma(
        &ma2.zip(ma).map(|(x, y)| 2.0 * x - y).collect::<Vec<f64>>(),
        (window as f64).sqrt().floor() as usize,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Volatility Index Dynamic Average (VIDYA)
///
/// A type of moving average that uses a combination of short-term and long-term
/// volatility measures to create a dynamic and responsive indicator.
///
/// VIDYA = α * (STV / LTV) * (Current Price - Previous VIDYA) + Previous VIDYA
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::vidya(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn vidya<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let alpha = 2.0 / (window + 1) as f64;
    let std5 = _std_dev(data, 5).collect::<Vec<f64>>();
    let std20 = sma(&std5, 20).collect::<Vec<f64>>();
    let offset = (5 - 1) + (20 - 1);
    iter::repeat(f64::NAN)
        .take(offset)
        .chain(
            izip!(
                std20.into_iter().skip(20 - 1),
                std5.into_iter().skip(20 - 1),
                data.iter().skip(offset)
            )
            .scan(0.0, move |state, (s20, s5, d)| {
                *state = alpha * (s5 / s20) * (d.to_f64().unwrap() - *state) + *state;
                Some(*state)
            }),
            // TODO: investigate why faster with collect().iter() than without, same for: fbands
        )
        .collect::<Vec<f64>>()
        .into_iter()
}

pub(crate) fn _cmo<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::once((0.0, 0.0))
        .chain(data.windows(2).scan((0.0, 0.0), |state, pair| {
            let (prev, curr) = (pair[0].to_f64().unwrap(), pair[1].to_f64().unwrap());
            let gain = state.0 + f64::max(0.0, curr - prev);
            let loss = state.1 + f64::max(0.0, prev - curr);
            *state = (gain, loss);
            Some((gain, loss))
        }))
        .collect::<Vec<(f64, f64)>>()
        .windows(window + 1)
        .map(|w| {
            let gain_sum = w[w.len() - 1].0 - w[0].0;
            let loss_sum = w[w.len() - 1].1 - w[0].1;
            (gain_sum - loss_sum) / (gain_sum + loss_sum)
        })
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Variable Moving Average (VMA)
///
/// A dynamic moving average that is calculated based on the Chande Momentum Oscillator values.
/// The VMA is used to smooth out the CMO values and provide a more stable signal.
/// Similar to VIDYA except it uses CMO rather than standard deviation values
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::vma(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn vma<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let alpha = 2.0 / (window + 1) as f64;
    let cmo_win = 9; // maybe make this configurable?
    let vi = _cmo(data, cmo_win);
    iter::repeat(f64::NAN).take(window.max(cmo_win)).chain(
        izip!(vi, data.iter().filter_map(|x| x.to_f64()).skip(cmo_win))
            .scan(0.0, move |state, (vi, d)| {
                *state = alpha * vi.abs() * (d - *state) + *state;
                Some(*state)
            })
            .skip(window.max(cmo_win) - cmo_win),
    )
}

/// Linear Regression Forecast (aka Time Series Forecast aka Least Squares Moving Average)
///
/// A type of moving average that incorporates the slope and intercept of a linear regression
/// line to make predictions.
///
/// ## Sources
///
/// [[1]](https://quantstrategy.io/blog/what-is-tsf-understanding-time-series-forecast-indicator/)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::lrf(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn lrf<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let x_sum = (window * (window + 1)) as f64 / 2.0;
    let x2_sum: f64 = x_sum * (2 * window + 1) as f64 / 3.0;
    let divisor = window as f64 * x2_sum - x_sum * x_sum;
    let indices: Vec<f64> = (1..=window).map(|x| x as f64).collect();

    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mut y_sum = 0.0;
            let mut xy_sum = 0.0;
            for (count, val) in indices.iter().zip(w.iter()) {
                y_sum += val.to_f64().unwrap();
                xy_sum += count * val.to_f64().unwrap();
            }
            let m = (window as f64 * xy_sum - x_sum * y_sum) / divisor;
            let b = (y_sum * x2_sum - x_sum * xy_sum) / divisor;
            m * window as f64 + b
        }))
}

/// Triangular moving average
///
/// A type of moving average that uses a triangular weighting scheme to calculate the
/// average value of a time series. It gives more weight to the middle values of the
/// time series and less weight to the values at the edges.
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::trima(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn trima<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let win1 = window.div_ceil(2);
    let win2 = if window & 2 == 0 { win1 + 1 } else { win1 };
    sma(&sma(data, win1).collect::<Vec<f64>>(), win2)
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Zero Lag moving average
///
/// Developed by John Ehlers and Ric Way. A type of moving average that aims to eliminate
/// the lag associated with traditional moving averages.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::zlma(&[1.0,2.0,3.0,4.0,5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn zlma<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let lag = (window - 1) / 2;
    iter::repeat(f64::NAN).take(lag).chain(
        ewma(
            &data
                .iter()
                .zip(data[lag..].iter())
                .map(|(prev, curr)| 2.0 * curr.to_f64().unwrap() - prev.to_f64().unwrap())
                .collect::<Vec<f64>>(),
            window,
        )
        .collect::<Vec<f64>>(),
    )
}

/// Kernel Regression
///
/// A non-parametric technique used to estimate the conditional expectation of a
/// random variable. The goal is to find a non-linear relationship between a pair of
/// random variables, denoted as X and Y.
///
/// Note: This implementation leverages a Gaussian kernel and currently only considers
/// historical data and a backwindow of 255 datapoints when computing value.
///
/// ## Sources
///
/// [[1]](https://www.stat.cmu.edu/~ryantibs/advmethods/notes/kernel.pdf)
/// [[2]](https://mccormickml.com/2014/02/26/kernel-regression/)
///
/// This only considers historical data and a backwindow of 255 datapoints.
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::kernel(&[1.0, 2.0, 3.0, 4.0, 5.0], 3.0).collect::<Vec<f64>>();
/// ```
pub fn kernel<T: ToPrimitive>(data: &[T], sigma: f64) -> impl Iterator<Item = f64> + '_ {
    let beta = 1.0 / (2.0 * sigma * sigma);
    let window = 255;
    let weights = (0..=window)
        .map(|x| (-beta * (x * x) as f64).exp())
        .collect::<Vec<f64>>();
    (0..data.len()).map(move |i| {
        let mut sum: f64 = 0.0;
        let mut sumw: f64 = 0.0;
        // for (j, val) in data.iter().take(i+1).enumerate() {
        //     // gaussian kernel
        //     let w = (-beta * (i as f64 - j as f64).powi(2)).exp();
        for (w, val) in weights
            .iter()
            .zip(data[..=i].iter().rev().take(std::cmp::min(i + 1, window)))
        {
            sum += w * val.to_f64().unwrap();
            sumw += w;
        }
        sum / sumw
    })
}

/// Kaufman Adaptive Moving Average (KAMA)
///
/// Similar to VIDYA, in that it uses two smoothing constants. Computes an Efficiency Ratio to
/// adapt the moving average to price trends.
///
/// ## Sources
///
/// [[1]](https://www.marketvolume.com/technicalanalysis/kama.asp)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::kama(&[1.0, 2.0, 3.0, 4.0, 5.0], 3, Some(2), Some(30)).collect::<Vec<f64>>();
/// ```
pub fn kama<T: ToPrimitive>(
    data: &[T],
    window: usize,
    short: Option<usize>,
    long: Option<usize>,
) -> impl Iterator<Item = f64> + '_ {
    let short = 2.0 / (short.unwrap_or(2) + 1) as f64;
    let long = 2.0 / (long.unwrap_or(30) + 1) as f64;

    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).scan(0.0, move |state, x| {
            let er = (x[x.len() - 1].to_f64().unwrap() - x[0].to_f64().unwrap()).abs()
                / (x.windows(2).fold(0.0, |acc, pair| {
                    acc + (pair[0].to_f64().unwrap() - pair[1].to_f64().unwrap()).abs()
                }));
            let alpha = (er * (short - long) + long).powi(2);
            *state = alpha * (x[x.len() - 1].to_f64().unwrap() - *state) + *state;
            Some(*state)
        }))
}

/// Arnaud Legoux Moving Average (ALMA)
///
/// Design to use Gaussian distribution that is shifted with a calculated offset in order
/// for the average to be biased towards more recent days
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/support/solutions/43000594683-arnaud-legoux-moving-average/)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::alma(&[1.0, 2.0, 3.0, 4.0, 5.0], 3, 2.0, Some(0.5)).collect::<Vec<f64>>();
/// ```
pub fn alma<T: ToPrimitive>(
    data: &[T],
    window: usize,
    sigma: f64,
    mu: Option<f64>,
) -> impl Iterator<Item = f64> + '_ {
    let mu = mu.unwrap_or(0.85) * (window - 1) as f64;
    let weights = (0..window)
        .map(|x| (-1.0 * (x as f64 - mu).powi(2) / (2.0 * (window as f64 / sigma).powi(2))).exp())
        .collect::<Vec<f64>>();
    iter::repeat(f64::NAN)
        .take(window)
        .chain((window..data.len()).map(move |i| {
            let mut sum: f64 = 0.0;
            let mut sumw: f64 = 0.0;
            for (idx, w) in weights.iter().enumerate() {
                sum += w * data[i - idx].to_f64().unwrap();
                sumw += w;
            }
            sum / sumw
        }))
}

/// McGinley Dynamic Moving Average
///
/// Takes into account speed changes in a market to show a smoother,
/// more responsive, moving average line.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/m/mcginley-dynamic.asp)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::mdma(&[1.0, 2.0, 3.0, 4.0, 5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn mdma<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let k = 0.6;
    iter::once(data[0].to_f64().unwrap()).chain(data[1..].iter().scan(
        data[0].to_f64().unwrap(),
        move |state, x| {
            *state += (x.to_f64().unwrap() - *state)
                / (k * window as f64 * (x.to_f64().unwrap() / *state).powi(4));
            Some(*state)
        },
    ))
}

/// Holt-Winter Moving Average
///
/// Applies exponential smoothing across 3 factors: value, trend, and seasonality.
///
/// ## Sources
///
/// [[1]](https://orangematter.solarwinds.com/2019/12/15/holt-winters-forecasting-simplified/)
/// [[2]](https://medium.com/analytics-vidhya/a-thorough-introduction-to-holt-winters-forecasting-c21810b8c0e6)
/// [[3]](https://www.mql5.com/en/code/20856)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::hwma(&[1.0, 2.0, 3.0, 4.0, 5.0],
///              Some(0.2), Some(0.1), None).collect::<Vec<f64>>();
/// ```
pub fn hwma<T: ToPrimitive>(
    data: &[T],
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
) -> impl Iterator<Item = f64> + '_ {
    let alpha = alpha.unwrap_or(0.2).clamp(0.0, 1.0);
    let beta = beta.unwrap_or(0.1).clamp(0.0, 1.0);
    let gamma = gamma.unwrap_or(0.1).clamp(0.0, 1.0);

    data.iter()
        .scan((data[0].to_f64().unwrap(), 0.0, 0.0), move |state, x| {
            let avg =
                (1.0 - alpha) * (state.0 + state.1 + 0.5 * state.2) + alpha * x.to_f64().unwrap();
            let trend = (1.0 - beta) * (state.1 + state.2) + beta * (avg - state.0);
            let season = (1.0 - gamma) * state.2 + gamma * (trend - state.1);
            *state = (avg, trend, season);
            Some(avg + trend + 0.5 * season)
        })
}

/// Fibonacci's Weighted Moving Average
///
/// Incorporates Fibonacci ratios into its calculation, assigning varying weights to
/// data points, enhancing its responsiveness.
///
/// ## Sources
///
/// [[1]](https://blog.xcaldata.com/confirming-market-trends-with-fibonacci-weighted-moving-average-fwma/)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::fwma(&[1.0, 2.0, 3.0, 4.0, 5.0], 3).collect::<Vec<f64>>();
/// ```
pub fn fwma<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let mut denom = 2.0;
    let weights = iter::repeat(1.0)
        .take(2)
        .chain((0..window - 2).scan((1.0, 1.0), |state, _| {
            *state = (state.1, state.0 + state.1);
            denom += state.1;
            Some(state.1)
        }))
        .collect::<Vec<f64>>()
        .iter()
        .map(|i| i / denom)
        .collect::<Vec<f64>>();
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            w.iter()
                .zip(weights.iter())
                .map(|(value, weight)| value.to_f64().unwrap() * weight)
                .sum()
        }))
}

/// Ehler's Super Smoother Filter
///
/// Aims to eliminate short-term fluctuations while reacting quickly.
/// The 2-pole logic balances smoothing and responsiveness while the 3-pole option prioritises
/// smoothing more.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/script/VdJy0yBJ-Ehlers-Super-Smoother-Filter/)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::ssf(&[1.0, 2.0, 3.0, 4.0, 5.0], 3, Some(false)).collect::<Vec<f64>>();
/// ```
pub fn ssf<T: ToPrimitive>(
    data: &[T],
    window: usize,
    tri_poles: Option<bool>,
) -> Box<dyn Iterator<Item = f64> + '_> {
    let tri_poles = tri_poles.unwrap_or(false);
    if tri_poles {
        let x = PI / window as f64;
        let y = (-x).exp();
        let y2 = y * y;
        let z = 2.0 * y * (3.0_f64.sqrt() * x).cos();
        let d = y2 * y2;
        let c = -y2 * (1.0 + z);
        let b = y2 + z;
        let a = 1.0 - b - c - d;
        Box::new(iter::repeat(f64::NAN).take(3).chain(data[3..].iter().scan(
            (
                data[0].to_f64().unwrap(),
                data[1].to_f64().unwrap(),
                data[2].to_f64().unwrap(),
            ),
            move |state, x| {
                *state = (
                    state.1,
                    state.2,
                    a * x.to_f64().unwrap() + b * state.2 + c * state.1 + d * state.0,
                );
                Some(state.2)
            },
        )))
    } else {
        let x = PI * 2.0_f64.sqrt() / window as f64;
        let y = (-x).exp();
        let c = -y * y;
        let b = 2.0 * y * x.cos();
        let a = 1.0 - b - c;
        Box::new(iter::repeat(f64::NAN).take(2).chain(data[2..].iter().scan(
            (data[0].to_f64().unwrap(), data[1].to_f64().unwrap()),
            move |state, x| {
                *state = (state.1, a * x.to_f64().unwrap() + b * state.1 + c * state.0);
                Some(state.1)
            },
        )))
    }
}

/// MESA Adaptive Moving Average
///
/// Built to adjust and adapt to ever-changing market dynamics. Uses a Hilbert Transform
/// Discriminator, which is a linear operator that imparts a 90-degree phase shift to frequency
/// components of an equation to keep up with market movements. Produces an additional
/// Following Adaptive Moving Average (FAMA) that undergoes vertical movements with a lag.
///
/// ## Sources
///
/// [[1]](https://www.mesasoftware.com/papers/MAMA.pdf)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::mama(&[1.0, 2.0, 3.0, 4.0, 5.0], None, Some(0.05)).collect::<Vec<(f64,f64)>>();
/// ```
pub fn mama<T: ToPrimitive>(
    data: &[T],
    fast: Option<f64>,
    slow: Option<f64>,
) -> impl Iterator<Item = (f64, f64)> + '_ {
    let fast = fast.unwrap_or(0.5);
    let slow = slow.unwrap_or(0.05);

    let sm = iter::repeat(0.0)
        .take(3)
        .chain(data.windows(4).map(|w| {
            (4.0 * w[3].to_f64().unwrap()
                + 3.0 * w[2].to_f64().unwrap()
                + 2.0 * w[1].to_f64().unwrap()
                + w[0].to_f64().unwrap())
                / 10.0
        }))
        .collect::<Vec<f64>>();
    let mut dtrend: VecDeque<f64> = [0.0; 7].into();

    let mut i1: VecDeque<f64> = [0.0; 7].into();
    let mut q1: VecDeque<f64> = [0.0; 7].into();

    iter::repeat((f64::NAN, f64::NAN))
        .take(6)
        .chain(
            (6..data.len()).scan((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), move |state, i| {
                // state = (mama, fama, period, phase, re, im, i2, q2)
                let tweak = 0.075 * state.2 + 0.54;

                dtrend.pop_back();
                dtrend.push_front(
                    ((0.0962 * sm[i]) + (0.5769 * sm[i - 2])
                        - (0.5769 * sm[i - 4])
                        - (0.0962 * sm[i - 6]))
                        * tweak,
                );

                // Compute InPhase and Quadrature components
                q1.pop_back();
                q1.push_front(
                    ((0.0962 * dtrend[0]) + (0.5769 * dtrend[2])
                        - (0.5769 * dtrend[4])
                        - (0.0962 * dtrend[6]))
                        * tweak,
                );
                i1.pop_back();
                i1.push_front(dtrend[3]);

                // Advance the phase of I1 and Q1 by 90 degrees}
                let ji =
                    ((0.0962 * i1[0]) + (0.5769 * i1[2]) - (0.5769 * i1[4]) - (0.0962 * i1[6]))
                        * tweak;
                let jq =
                    ((0.0962 * q1[0]) + (0.5769 * q1[2]) - (0.5769 * q1[4]) - (0.0962 * q1[6]))
                        * tweak;

                // Phasor addition for 3 bar averaging
                // Smooth the I and Q components before applying the discriminator
                let i2 = 0.2 * (i1[0] - jq) + (0.8 * state.6);
                let q2 = 0.2 * (q1[0] + ji) + (0.8 * state.7);

                // Homodyne Discriminator
                let re = 0.2 * ((i2 * state.6) + (q2 * state.7)) + (0.8 * state.4);
                let im = 0.2 * ((i2 * state.7) - (q2 * state.6)) + (0.8 * state.5);

                let mut pd = if im != 0.0 && re != 0.0 {
                    360.0 / (im / re).atan()
                } else {
                    0.0
                }
                .clamp(0.67 * state.2, 1.5 * state.2)
                .clamp(6.0, 50.0);
                pd = (0.2 * pd) + (0.8 * state.2);

                let phase = if i1[0] != 0.0 {
                    (q1[0] / i1[0]).atan()
                } else {
                    0.0
                };
                let delta_phase = (state.3 - phase).max(1.0);

                let alpha = (fast / delta_phase).max(slow);
                let mama = alpha * data[i].to_f64().unwrap() + (1.0 - alpha) * state.0;
                *state = (
                    mama,
                    0.5 * alpha * mama + (1.0 - 0.5 * alpha) * state.1,
                    pd,
                    phase,
                    re,
                    im,
                    i2,
                    q2,
                );
                Some((state.0, state.1))
            }),
        )
}

/// T3 Moving Average
///
/// Another moving average leveraging EMA of EMAs to reduce noise.
///
/// ## Sources
///
/// [[1]](https://theforexgeek.com/t3-moving-average/)
///
/// # Examples
///
/// ```
/// use traquer::smooth;
///
/// smooth::t3(
///     &[1.0,2.0,3.0,4.0,5.0,2.0,3.0,4.0,2.0,3.0,4.0,2.0,3.0,4.0],
///     3, None).collect::<Vec<f64>>();
/// ```
pub fn t3<T: ToPrimitive>(
    data: &[T],
    window: usize,
    factor: Option<f64>,
) -> impl Iterator<Item = f64> + '_ {
    let factor = factor.unwrap_or(0.7).clamp(0.0, 1.0);
    let ma = ewma(data, window).collect::<Vec<f64>>();
    let ma2 = ewma(&ma[window - 1..], window).collect::<Vec<f64>>();
    let ma3 = ewma(&ma2[window - 1..], window).collect::<Vec<f64>>();
    let ma4 = ewma(&ma3[window - 1..], window).collect::<Vec<f64>>();
    let ma5 = ewma(&ma4[window - 1..], window).collect::<Vec<f64>>();
    let ma6 = ewma(&ma5[window - 1..], window).collect::<Vec<f64>>();
    let c1 = -factor.powi(3);
    let c2 = 3.0 * factor.powi(2) + 3.0 * factor.powi(3);
    let c3 = -3.0 * factor - 6.0 * factor.powi(2) - 3.0 * factor.powi(3);
    let c4 = 1.0 + 3.0 * factor + 3.0 * factor.powi(2) + factor.powi(3);

    iter::repeat(f64::NAN).take((window - 1) * 2).chain(
        izip!(
            ma3,
            iter::repeat(f64::NAN).take(window - 1).chain(ma4),
            iter::repeat(f64::NAN).take((window - 1) * 2).chain(ma5),
            iter::repeat(f64::NAN).take((window - 1) * 3).chain(ma6)
        )
        .map(move |(e3, e4, e5, e6)| c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3),
    )
}
