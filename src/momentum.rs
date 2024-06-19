//! Momentum Indicators
//!
//! Provides technical indicators that measures the rate of change or speed of price
//! movement of a security. In the context of this library, these indicators are typically
//! range bound and/or centred around zero. These often begin to show trend the larger the
//! smoothing.

use std::iter;

use itertools::izip;

use crate::smooth;
use crate::volatility::_true_range;

/// Relative Strength Index
///
/// Calculated by comparing the average gain of up days to the average loss of down days
/// over a specified period. Shows the magnitude of recent price changes to determine
/// overbought or oversold conditions.
///
/// ## Usage
///
/// Usually, a value above 70 suggests overbought and a value below 30, oversold.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/r/rsi.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::rsi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn rsi(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let (gain, loss): (Vec<f64>, Vec<f64>) = data
        .windows(2)
        .map(|pair| {
            (
                f64::max(0.0, pair[1] - pair[0]),
                f64::min(0.0, pair[1] - pair[0]).abs(),
            )
        })
        .unzip();
    iter::once(f64::NAN).chain(
        smooth::wilder(&gain, window)
            .zip(smooth::wilder(&loss, window))
            .map(|(g, l)| 100.0 * g / (g + l))
            .collect::<Vec<f64>>(),
    )
}

/// Moving Average Convergence/Divergence
///
/// Shows the convergence and divergence of the two moving averages, indicating changes in
/// the strength and direction of the trend. When the MACD crosses above the signal line,
/// it's a bullish signal, indicating a potential uptrend.
///
/// ## Usage
///
/// An increasing value suggests a stronger uptrend. Often paired with signal line to suggests buy/sell.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/m/macd.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::macd(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn macd(close: &[f64], short: usize, long: usize) -> impl Iterator<Item = f64> + '_ {
    let short_ma = smooth::ewma(close, short);
    let long_ma = smooth::ewma(close, long);
    short_ma.zip(long_ma).map(|(x, y)| x - y)
}

/// Chande Momentum Oscillator
///
/// The CMO oscillates between +100 and -100, with high values indicating strong upward
/// momentum and low values indicating strong downward momentum. The indicator is
/// calculated by summing up the positive and negative price changes over a specified
/// period, then dividing the result by the sum of the absolute values of all price
/// changes over the same period.
///
/// ## Usage
///
/// A reading above 50 indicates strong bullish momentum, while a reading below -50
/// suggests strong bearish momentum.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/c/chandemomentumoscillator.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::cmo(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cmo(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN)
        .take(window)
        .chain(smooth::_cmo(data, window).map(|x| x * 100.0))
}

/// Chande Forecast Oscillator
///
/// An indicator that attempts to forecast the future direction of the market by
/// analyzing the relationship between the current price and the price a certain number
/// of periods ago.
///
/// The resulting value is then multiplied by 100 to create an oscillator that ranges
/// from -100 to +100.
///
/// ## Usage
///
/// A value greater than 0, suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.stockmaniacs.net/chande-forecast-oscillator/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::cfo(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cfo(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    smooth::lrf(data, window)
        .zip(data)
        .map(|(tsf, x)| 100.0 * (x - tsf) / x)
}

/// Elder Ray
///
/// The Elder Ray Index consists of two components:
///
/// Bull Power
///   - This measures the ability of buyers to push the price up.
///   - It's calculated by subtracting the EWMA from the high price.
///
/// Bear Power
///   - This measures the ability of sellers to push the price down.
///   -  It's calculated by subtracting the EWMA from the low price.
///
/// ## Usage
///
/// Increasing bull and bear values above 0 suggest a stronger uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/articles/trading/03/022603.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::elder_ray(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<(f64,f64)>>();
///
/// ```
pub fn elder_ray<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let close_ma = smooth::ewma(close, window);
    izip!(high, low, close_ma).map(|(h, l, c)| (h - c, l - c))
}

/// Williams Alligator
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/articles/trading/072115/exploring-williams-alligator-indicator.asp)
pub fn alligator(_data: &[f64]) {}

/// Williams Percent Range
///
/// Measure the level of a security's close price in relation to its high-low range over a
/// specified period.
///
/// W%R = (Highest High - Close) / (Highest High - Lowest Low) * -100
///
/// ## Usage
///
/// Typically, a value above -20 suggests overbought and below -80, oversold.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/w/williamsr.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::wpr(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn wpr<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        izip!(
            high.windows(window),
            low.windows(window),
            &close[(window - 1)..]
        )
        .map(|(h, l, c)| {
            let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
            let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
            -100.0 * ((hh - c) / (hh - ll))
        }),
    )
}

/// Percent Price Oscillator
///
/// Measure the difference between two moving averages as a percentage of the larger
/// moving average. Effectively same as Chande's Range Action Verification Index.
///
/// ## Usage
///
/// A value above zero suggests an uptrend
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::ppo(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn ppo(data: &[f64], short: usize, long: usize) -> impl Iterator<Item = f64> + '_ {
    let short_ma = smooth::ewma(data, short);
    let long_ma = smooth::ewma(data, long);
    short_ma.zip(long_ma).map(|(x, y)| 100.0 * (x / y - 1.0))
}

/// Absolute Price Oscillator
///
/// Measure the difference between two moving averages.
///
/// ## Usage
///
/// A value above zero suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::apo(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn apo(data: &[f64], short: usize, long: usize) -> impl Iterator<Item = f64> + '_ {
    let short_ma = smooth::ewma(data, short);
    let long_ma = smooth::ewma(data, long);
    short_ma.zip(long_ma).map(|(x, y)| x - y)
}

/// Price Momentum Oscillator
///
/// A double smoothed version of ROC designed to track changes in a trend strength
///
/// ## Usage
///
/// A value above zero suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.marketvolume.com/technicalanalysis/pmo.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::pmo(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, 3).collect::<Vec<f64>>();
///
/// ```
pub fn pmo(data: &[f64], win1: usize, win2: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(win1).chain(
        smooth::ewma(
            &smooth::ewma(
                &data
                    .windows(2)
                    .map(|pair| 1000.0 * (pair[1] / pair[0] - 1.0))
                    .collect::<Vec<f64>>(),
                win1,
            )
            .collect::<Vec<f64>>()[win1 - 1..],
            win2,
        )
        .collect::<Vec<f64>>(),
    )
}

/// Ultimate Oscillator
///
/// A technical indicator that uses the weighted average of three different time periods
/// to reduce the volatility and false transaction signals.
///
/// ## Usage
///
/// Typically, a value above 70 suggests overbought and a value below 30, oversold.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/u/ultimateoscillator.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::ultimate(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2, 4, 8).collect::<Vec<f64>>();
///
/// ```
pub fn ultimate<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    win1: usize,
    win2: usize,
    win3: usize,
) -> impl Iterator<Item = f64> + 'a {
    let bp_tr_vals = izip!(
        &high[1..],
        &low[1..],
        &close[..close.len() - 1],
        &close[1..],
    )
    .map(|(h, l, prevc, c)| {
        (
            c - f64::min(*l, *prevc),
            f64::max(*h, *prevc) - f64::min(*l, *prevc),
        )
    })
    .collect::<Vec<(f64, f64)>>();
    iter::repeat(f64::NAN).take(win3).chain(
        bp_tr_vals
            .windows(win3)
            .map(|w| {
                let (bp_sum1, tr_sum1) = w
                    .iter()
                    .skip(win3 - win1)
                    .fold((0.0, 0.0), |acc, (bp, tr)| (acc.0 + bp, acc.1 + tr));
                let (bp_sum2, tr_sum2) = w
                    .iter()
                    .skip(win3 - win2)
                    .fold((0.0, 0.0), |acc, (bp, tr)| (acc.0 + bp, acc.1 + tr));
                let (bp_sum3, tr_sum3) = w
                    .iter()
                    .fold((0.0, 0.0), |acc, (bp, tr)| (acc.0 + bp, acc.1 + tr));
                100.0 * (bp_sum1 / tr_sum1 * 4.0 + bp_sum2 / tr_sum2 * 2.0 + bp_sum3 / tr_sum3)
                    / (4.0 + 2.0 + 1.0)
            })
            .collect::<Vec<f64>>(),
    )
}

/// Pretty Good Oscillator
///
/// Combines moving averages and the Average True Range (ATR) to create an oscillator
/// that oscillates around a centerline
///
/// ## Usage
///
/// Typically, a value above 3 suggests overbought and a value below -3, oversold.
///
/// ## Sources
///
/// [[1]](https://library.tradingtechnologies.com/trade/chrt-ti-pretty-good-oscillator.html)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::pgo(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn pgo<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let tr = _true_range(high, low, close).collect::<Vec<f64>>();
    let atr = iter::once(f64::NAN).chain(smooth::ewma(&tr, window));
    let sma_close = smooth::sma(close, window);
    izip!(close, sma_close, atr)
        .map(|(c, c_ma, tr_ma)| (c - c_ma) / tr_ma)
        .collect::<Vec<f64>>()
        .into_iter()
}

pub(crate) fn _swing<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    limit: f64,
) -> impl Iterator<Item = f64> + 'a {
    izip!(
        &open[..open.len() - 1],
        &open[1..],
        &high[1..],
        &low[1..],
        &close[..close.len() - 1],
        &close[1..],
    )
    .map(move |(prevo, o, h, l, prevc, c)| {
        let r1 = (h - prevc).abs();
        let r2 = (l - prevc).abs();
        let r3 = h - l;
        let r4 = (prevc - prevo).abs() / 4.0;
        let max_r = r1.max(r2).max(r3);
        let r = if r1 == max_r {
            r1 - r2 / 2.0 + r4
        } else if r2 == max_r {
            r2 - r1 / 2.0 + r4
        } else {
            r3 + r4
        };
        // does not use formula described in investopedia as it appears to be wrong?
        // it seems to overweight previous period's values
        ((c - prevc + (c - o) / 2.0 + (prevc - prevo) / 4.0) / r) * 50.0 * r1.max(r2) / limit
    })
}

/// Swing Index
///
/// Calculates the strength of price movement and predicts potential trend reversal.
///
/// ## Usage
///
/// A value above 0 suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/a/asi.asp)
/// [[2]](https://quantstrategy.io/blog/accumulative-swing-index-how-to-trade/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::si(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     0.5).collect::<Vec<f64>>();
///
/// ```
pub fn si<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    limit: f64,
) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(_swing(open, high, low, close, limit))
}

/// Triple Exponential Average
///
/// Indicator to show the percentage change in a moving average that has been smoothed
/// exponentially three times.
///
/// ## Usage
///
/// A value above 0 suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/t/trix.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::trix(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,5.0,2.0],
///     3).collect::<Vec<f64>>();
///
/// ```
pub fn trix(close: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let ema3 = smooth::ewma(
        &smooth::ewma(
            &smooth::ewma(close, window).collect::<Vec<f64>>()[window - 1..],
            window,
        )
        .collect::<Vec<f64>>()[window - 1..],
        window,
    )
    .collect::<Vec<f64>>();
    iter::repeat(f64::NAN).take((window - 1) * 2 + 1).chain(
        ema3.iter()
            .zip(&ema3[1..])
            .map(|(prev, curr)| 100.0 * (curr - prev) / prev)
            .collect::<Vec<f64>>(),
    )
}

/// Trend Intensity Index
///
/// Uses RSI principles but applies them to closing price deviations instead of the closing prices
///
/// ## Usage
///
/// Typically, a value above 80 suggests overbought and a value below 20, oversold.
///
/// ## Sources
///
/// [[1]](https://www.marketvolume.com/technicalanalysis/trendintensityindex.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::tii(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn tii(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let win = window.div_ceil(2);
    iter::repeat(f64::NAN).take(win - 1).chain(
        smooth::sma(data, window)
            .zip(data)
            .map(|(avg, actual)| {
                let dev: f64 = actual - avg;
                (dev.max(0.0), dev.min(0.0).abs())
            })
            .collect::<Vec<(f64, f64)>>()
            .windows(win)
            .map(|w| {
                let mut sd_pos = 0.0;
                let mut sd_neg = 0.0;
                for (pos_dev, neg_dev) in w {
                    sd_pos += pos_dev;
                    sd_neg += neg_dev;
                }
                100.0 * sd_pos / (sd_pos + sd_neg)
            })
            .collect::<Vec<f64>>(),
    )
}

/// Stochastic Oscillator
///
/// Compares a security’s closing price to a range of its highest highs and lowest lows
/// over a specific time period.
///
/// ## Usage
///
/// Typically, a value above 80 suggests overbought and a value below 20, oversold.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/articles/technical/073001.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::stochastic(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn stochastic<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let smooth_win = 3;
    let fast_k = smooth::sma(
        &izip!(
            high.windows(window),
            low.windows(window),
            &close[(window - 1)..]
        )
        .map(|(h, l, c)| {
            let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
            let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
            100.0 * (c - ll) / (hh - ll)
        })
        .collect::<Vec<f64>>(),
        smooth_win,
    )
    .collect::<Vec<f64>>();
    let k = smooth::sma(&fast_k[smooth_win - 1..], smooth_win).collect::<Vec<f64>>();
    izip!(
        iter::repeat(f64::NAN).take(window - 1).chain(fast_k),
        iter::repeat(f64::NAN)
            .take((window - 1) + (smooth_win - 1))
            .chain(k)
    )
}

fn _stc(series: &[f64], window: usize, smooth: usize) -> impl Iterator<Item = f64> + '_ {
    smooth::wilder(
        &series
            .windows(window)
            .map(|w| {
                let mut hh = f64::NAN;
                let mut ll = f64::NAN;
                for x in w {
                    hh = hh.max(*x);
                    ll = ll.min(*x);
                }
                100.0 * (w.last().unwrap() - ll) / (hh - ll)
            })
            .collect::<Vec<f64>>(),
        smooth,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Schaff Trend Cycle
///
/// A modified version of the Moving Average Convergence Divergence. It aims to improve
/// upon traditional moving averages (MAs) by incorporating cycle analysis.
///
/// ## Usage
///
/// Typically a value above 75 suggests overbought and a value below 25, oversold.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/articles/forex/10/schaff-trend-cycle-indicator.asp)
/// [[2]](https://quantstrategy.io/blog/understanding-the-schaff-trend-cycle-stc-indicator-a-powerful-technical-analysis-tool/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::stc(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,6.0,1.0],
///     2, 3, 5).collect::<Vec<f64>>();
///
/// ```
pub fn stc(
    close: &[f64],
    window: usize,
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
    let smooth = 2;
    iter::repeat(f64::NAN)
        .take((long - 1) + (window - 1) * 2 + (smooth - 1) * 2)
        .chain(
            _stc(
                &_stc(
                    &macd(close, short, long)
                        .skip(long - 1)
                        .collect::<Vec<f64>>(),
                    window,
                    smooth,
                )
                .skip(smooth - 1)
                .collect::<Vec<f64>>(),
                window,
                smooth,
            )
            .skip(smooth - 1)
            .collect::<Vec<f64>>(),
        )
}

/// Relative Vigor
///
/// Measures the strength of a trend by comparing a security’s closing price to its trading range
/// while smoothing the results using a simple moving average.
///
/// ## Usage
///
/// A value above 0 suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/r/relative_vigor_index.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::relative_vigor(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn relative_vigor<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let close_open = open
        .iter()
        .zip(close)
        .map(|(o, c)| c - o)
        .collect::<Vec<f64>>();
    let high_low = high
        .iter()
        .zip(low)
        .map(|(h, l)| h - l)
        .collect::<Vec<f64>>();

    let numerator = close_open
        .windows(4)
        .map(|w| (w[3] + 2.0 * w[2] + 2.0 * w[1] + w[0]) / 6.0)
        .collect::<Vec<f64>>();
    let denominator = high_low
        .windows(4)
        .map(|w| (w[3] + 2.0 * w[2] + 2.0 * w[1] + w[0]) / 6.0)
        .collect::<Vec<f64>>();
    iter::repeat(f64::NAN).take(4 - 1).chain(
        smooth::sma(&numerator, window)
            .zip(smooth::sma(&denominator, window))
            .map(|(n, d)| n / d)
            .collect::<Vec<f64>>(),
    )
}

/// Elhers Fisher Transform
///
/// Converts price data into a Gaussian normal distribution.
/// Extreme readings (above +1 or below -1) may signal potential price reversals.
///
/// ## Usage
///
/// A value above 0 suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/f/fisher-transform.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::fisher(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn fisher<'a>(
    high: &'a [f64],
    low: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let hl2 = high
        .iter()
        .zip(low)
        .map(|(h, l)| (h + l) / 2.0)
        .collect::<Vec<f64>>();
    iter::repeat(f64::NAN).take(window - 1).chain(
        hl2.windows(window)
            .scan((0.0, 0.0), |state, w| {
                let mut hl_max: f64 = 0.0;
                let mut hl_min: f64 = f64::MAX;
                for &e in w {
                    hl_max = hl_max.max(e);
                    hl_min = hl_min.min(e);
                }
                let transform = (0.66
                    * ((w[window - 1] - hl_min) / (hl_max - hl_min).max(0.000001) - 0.5)
                    + 0.67 * state.0)
                    .clamp(-0.999999, 0.999999);
                let result = 0.5 * ((1.0 + transform) / (1.0 - transform)).ln() + 0.5 * state.1;
                *state = (transform, result);
                Some(state.1)
            })
            .collect::<Vec<f64>>(),
    )
}

/// Rainbow Oscillator
///
/// Based on multiple simple moving averages (SMAs). The highest high and lowest low
/// of these SMAs create high and low oscillator curves.
///
/// ## Usage
///
/// An oscillator value above 0 suggests an uptrend. A higher band value suggests instability.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/script/gWYg0ti0-Indicators-Rainbow-Charts-Oscillator-Binary-Wave-and-MAs/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::rainbow(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2, 3).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn rainbow(
    data: &[f64],
    window: usize,
    lookback: usize,
) -> impl Iterator<Item = (f64, f64)> + '_ {
    let mut smas = Vec::with_capacity(10);
    smas.push(smooth::sma(data, window).collect::<Vec<f64>>());
    for i in 1..10 {
        smas.push(smooth::sma(&smas[i - 1], window).collect::<Vec<f64>>());
    }
    iter::repeat((f64::NAN, f64::NAN))
        .take((window - 1) * 10)
        .chain(((window - 1) * 10..data.len()).map(move |i| {
            let mut total: f64 = 0.0;
            let mut hsma = f64::MIN;
            let mut lsma = f64::MAX;
            for sma in smas.iter() {
                let val = sma[i];
                total += val;
                hsma = hsma.max(val);
                lsma = lsma.min(val);
            }
            let mut hlookback = f64::MIN;
            let mut llookback = f64::MAX;
            ((i - (lookback - 1)).max(0)..=i).for_each(|x| {
                let val = data[x];
                hlookback = hlookback.max(val);
                llookback = llookback.min(val);
            });
            let osc = 100.0 * (data[i] - total / 10.0) / (hlookback - llookback).max(0.000001);
            let band = 100.0 * (hsma - lsma) / (hlookback - llookback).max(0.000001);
            (osc, band)
        }))
}

/// Coppock Curve
///
/// Calculated as a weighted moving average of the sum of two rate of change periods
///
/// ## Usage
///
/// A value above 0 suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/c/coppockcurve.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::coppock(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2, 3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn coppock(
    data: &[f64],
    window: usize,
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(long).chain(
        smooth::wma(
            &(long..data.len())
                .map(|x| 100.0 * (data[x] / data[x - short] + data[x] / data[x - long] - 2.0))
                .collect::<Vec<f64>>(),
            window,
        )
        .collect::<Vec<f64>>(),
    )
}

/// Rate of Change
///
/// Measures the percentage change in price between the current price
/// and the price a certain number of periods prior.
///
/// ## Usage
///
/// A value above 0 suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/p/pricerateofchange.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::roc(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn roc(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(window - 1).chain(
        data.windows(window)
            .map(|w| 100.0 * (w[w.len() - 1] / w[0] - 1.0)),
    )
}

/// Balance of Power
///
/// An oscillator that measures the strength of buying and selling pressure
///
/// ## Usage
///
/// A value above 0 suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/support/solutions/43000589100-balance-of-power-bop/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::bal_power(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<f64>>();
///
/// ```
pub fn bal_power<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    smooth::ewma(
        &izip!(open, high, low, close)
            .map(|(o, h, l, c)| (c - o) / (h - l))
            .collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Disparity Index
///
/// Measures the relative position of the most recent closing price to a selected
/// moving average as a percentage.
///
/// ## Usage
///
/// A value above 0 suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/d/disparityindex.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::disparity(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn disparity(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.iter()
        .zip(smooth::ewma(data, window))
        .map(|(x, ma)| 100.0 * (x - ma) / ma)
}

/// Quick Stick
///
/// Measures buying and selling pressure, taking an average of the difference between
/// closing and opening prices. When the price is closing lower than it opens, the
/// indicator moves lower. When the price is closing higher than the open,
/// the indicator moves up
///
/// ## Usage
///
/// A value greater than zero means that the majority of datapoints in period have been up,
/// indicating that buying pressure has been increasing.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/q/qstick.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::qstick(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn qstick<'a>(
    open: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let q = close
        .iter()
        .zip(open.iter())
        .map(|(c, o)| c - o)
        .collect::<Vec<f64>>();
    smooth::ewma(&q, window).collect::<Vec<f64>>().into_iter()
}

/// Centre of Gravity
///
/// Calculates the midpoint of a security's price action over a specified period.
///
/// ## Usage
///
/// Troughs suggest buy signal and peaks, sell.
///
/// ## Sources
///
/// [[1]](http://www.mesasoftware.com/papers/TheCGOscillator.pdf)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::cog(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cog(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let weights: Vec<f64> = (1..=window).map(|x| x as f64).collect();
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            -w.iter()
                .rev()
                .zip(weights.iter())
                .map(|(e, i)| e * i)
                .sum::<f64>()
                / w.iter().sum::<f64>()
        }))
}

/// Psychological Line
///
/// Based on the presumption that people will resist paying more for a share than others,
/// unless of course the share continues to move up. Conversely, people resist selling a
/// share for less than the price others have been getting for it, except if it continues to
/// decline.
///
/// Calculates a ratio based on the number of up bars (price higher than previous bar) over
/// a specified number of bars.
///
/// ## Usage
///
/// An increasing line suggests a stronger uptrend.
///
/// ## Sources
///
/// [[1]](https://tradingliteracy.com/psychological-line-indicator/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::psych(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn psych(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(window).chain(
        data.windows(2)
            .map(|pair| (pair[1] - pair[0]).signum().max(0.0))
            .collect::<Vec<f64>>()
            .windows(window)
            .map(|w| w.iter().sum::<f64>() * 100.0 / window as f64)
            .collect::<Vec<f64>>(),
    )
}

/// True Strength Index
///
/// Measures the strength of an asset or market's price movement over time as
/// well as any directions in that price.
///
/// ## Usage
///
/// Uptrends are denoted by tsi values crossing above signal line or centre line.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/t/tsi.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::tsi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     3, 6, 3).collect::<Vec<(f64,f64)>>();
///
/// ```
pub fn tsi(
    data: &[f64],
    short: usize,
    long: usize,
    signal: usize,
) -> impl Iterator<Item = (f64, f64)> + '_ {
    let diffs = data
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<f64>>();
    let long_diff = smooth::ewma(&diffs, long)
        .skip(long - 1)
        .collect::<Vec<f64>>();
    let pcds = smooth::ewma(&long_diff, short);
    let abs_long_diff = smooth::ewma(&diffs.iter().map(|x| x.abs()).collect::<Vec<f64>>(), long)
        .skip(long - 1)
        .collect::<Vec<f64>>();
    let abs_pcds = smooth::ewma(&abs_long_diff, short);
    let tsi = pcds
        .zip(abs_pcds)
        .map(|(pcd, apcd)| 100.0 * pcd / apcd)
        .collect::<Vec<f64>>();
    let signal = iter::repeat(f64::NAN)
        .take(short - 1)
        .chain(smooth::ewma(&tsi[short - 1..], signal));
    iter::repeat((f64::NAN, f64::NAN)).take(long).chain(
        tsi.iter()
            .zip(signal)
            .map(|(&x, y)| (x, y))
            .collect::<Vec<(f64, f64)>>(),
    )
}

/// Pring's Special K
///
/// The sum of several different weighted averages of different rate of change calculations.
/// Assumes that prices are revolving around the four-year business cycle.
///
/// ## Usage
///
/// Designed to peak and trough with the price at bull and bear market turning points.
///
/// ## Sources
///
/// [[1]](https://school.stockcharts.com/doku.php?id=technical_indicators:pring_s_special_k)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::special_k(&vec![1.0; 50]).collect::<Vec<f64>>();
///
/// ```
pub fn special_k(data: &[f64]) -> impl Iterator<Item = f64> + '_ {
    const ROCS: [usize; 12] = [10, 15, 20, 30, 50, 65, 75, 100, 195, 265, 390, 530];
    const PERIODS: [usize; 12] = [10, 10, 10, 15, 50, 65, 75, 100, 130, 130, 130, 195];
    const MULTIPLIERS: [f64; 12] = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    prings(data, &ROCS, &PERIODS, &MULTIPLIERS)
}

/// Pring's Know Sure Thing
///
/// Oscillator developed by Martin Pring to make rate-of-change readings easier for
/// traders to interpret.
///
/// ## Usage
///
/// Signals are generated when the KST crosses over the signal line, but traders also look for overbought or oversold conditions.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/k/know-sure-thing-kst.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::kst(&vec![1.0; 50], Some([10, 15, 25, 30]), None).collect::<Vec<f64>>();
///
/// ```
pub fn kst(
    data: &[f64],
    rocs: Option<[usize; 4]>,
    windows: Option<[usize; 4]>,
) -> impl Iterator<Item = f64> + '_ {
    let rocs = rocs.unwrap_or([10, 15, 20, 30]);
    let periods = windows.unwrap_or([10, 10, 10, 15]);
    const MULTIPLIERS: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    prings(data, &rocs, &periods, &MULTIPLIERS)
}

fn prings<'a, 'b>(
    data: &'a [f64],
    rocs: &'b [usize],
    periods: &'b [usize],
    multipliers: &'b [f64],
) -> impl Iterator<Item = f64> + 'a {
    let mut result = vec![0.0; data.len() - rocs[0] - (periods[0] - 1)];
    izip!(rocs, periods, multipliers).for_each(|(roc_win, ma_win, mult)| {
        let roc_win = roc_win + 1;
        let roc = data
            .windows(roc_win)
            .map(|w| 100.0 * (w[roc_win - 1] / w[0] - 1.0))
            .collect::<Vec<f64>>();
        if !roc.is_empty() {
            smooth::sma(&roc, *ma_win)
                .skip(ma_win - 1)
                .enumerate()
                .for_each(|(c, val)| {
                    result[c + ((roc_win - 1) - (rocs[0])) + (ma_win - periods[0])] += val * mult;
                });
        }
    });
    iter::repeat(f64::NAN)
        .take(rocs[0] + (periods[0] - 1))
        .chain(result)
}

/// Derivative Oscillator
///
/// An advanced version of the Relative Strength Index. It applies MACD Histogram principle
/// to the double smoothed RSI
///
/// ## Usage
///
/// A move above zero line suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.marketvolume.com/technicalanalysis/derivativeoscillator.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::derivative(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     2, 3, 2).collect::<Vec<f64>>();
///
/// ```
pub fn derivative(
    data: &[f64],
    win1: usize,
    win2: usize,
    signal: usize,
) -> impl Iterator<Item = f64> + '_ {
    let result = smooth::ewma(
        &smooth::ewma(&rsi(data, win1).skip(win1).collect::<Vec<_>>(), win1)
            .skip(win1 - 1)
            .collect::<Vec<_>>(),
        win2,
    )
    .collect::<Vec<_>>();
    let signal_line = smooth::sma(&result, signal);
    iter::repeat(f64::NAN).take(win1 + (win1 - 1)).chain(
        signal_line
            .zip(&result)
            .map(|(sig, val)| val - sig)
            .collect::<Vec<_>>(),
    )
}
