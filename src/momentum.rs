//! Momentum Indicators
//!
//! Provides technical indicators that measures the rate of change or speed of price
//! movement of a security. In the context of this library, these indicators are typically
//! range bound and/or centred around zero. These often begin to show trend the larger the
//! smoothing.

use std::iter;

use itertools::izip;
use num_traits::cast::ToPrimitive;

use crate::smooth;
use crate::trend::alligator;
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn rsi<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let (gain, loss): (Vec<f64>, Vec<f64>) = data
        .windows(2)
        .map(|pair| {
            (
                f64::max(0.0, pair[1].to_f64().unwrap() - pair[0].to_f64().unwrap()),
                f64::min(0.0, pair[1].to_f64().unwrap() - pair[0].to_f64().unwrap()).abs(),
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

/// Connors RSI
///
/// A variation of RSI that would adapt better to short-term changes. Designed to use
/// shorter lookback periods to capture more volatile and faster-moving action. Factors in
/// duration of trend and relative magnitude of change in addtion to RSI's price momentum.
///
/// ## Usage
///
/// Usually, a value above 90 suggests overbought and a value below 10, oversold.
///
/// ## Sources
///
/// [[1]](https://alvarezquanttrading.com/wp-content/uploads/2016/05/ConnorsRSIGuidebook.pdf)
/// [[2]](https://alvarezquanttrading.com/blog/connorsrsi-analysis/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::crsi(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 2, 6).collect::<Vec<f64>>();
///
/// ```
pub fn crsi<T: ToPrimitive>(
    data: &[T],
    rsi_win: usize,
    streak_win: usize,
    rank_win: usize,
) -> impl Iterator<Item = f64> + '_ {
    let rsi_series = rsi(data, rsi_win);
    let (streaks, returns): (Vec<f64>, Vec<f64>) = data
        .windows(2)
        .scan((0.0, 0.0_f64), |state, w| {
            let (w0, w1) = (w[0].to_f64().unwrap(), w[1].to_f64().unwrap());
            let streak = if w1 == w0 {
                0.0
            } else if (w1 - w0).signum() == state.1.signum() {
                state.0 + (w1 - w0).signum()
            } else {
                (w1 - w0).signum()
            };
            *state = (streak, (w1 - w0) / w0);
            Some(*state)
        })
        .unzip();
    let streak_rsi = iter::once(f64::NAN).chain(rsi(&streaks, streak_win));
    let rank = iter::repeat(f64::NAN)
        .take(rank_win)
        .chain(returns.windows(rank_win).map(|w| {
            let curr_ret = w[rank_win - 1];
            w.iter()
                .rev()
                .fold(0, |acc, &x| if x < curr_ret { acc + 1 } else { acc }) as f64
                / (rank_win - 1) as f64
                * 100.0
        }));
    izip!(rsi_series, streak_rsi, rank)
        .map(|(x, y, z)| (x + y + z) / 3.0)
        .collect::<Vec<f64>>()
        .into_iter()
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn macd<T: ToPrimitive>(
    close: &[T],
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cmo<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cfo<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    smooth::lrf(data, window)
        .zip(data)
        .map(|(tsf, x)| 100.0 * (x.to_f64().unwrap() - tsf) / x.to_f64().unwrap())
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<(f64,f64)>>();
///
/// ```
pub fn elder_ray<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let close_ma = smooth::ewma(close, window);
    izip!(high, low, close_ma).map(|(h, l, c)| (h.to_f64().unwrap() - c, l.to_f64().unwrap() - c))
}

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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn wpr<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        izip!(
            high.windows(window),
            low.windows(window),
            &close[(window - 1)..]
        )
        .map(|(h, l, c)| {
            let hh = h
                .iter()
                .fold(f64::NAN, |state, x| state.max(x.to_f64().unwrap()));
            let ll = l
                .iter()
                .fold(f64::NAN, |state, x| state.min(x.to_f64().unwrap()));
            -100.0 * ((hh - c.to_f64().unwrap()) / (hh - ll))
        }),
    )
}

/// Percent Price Oscillator
///
/// Measure the difference between two moving averages as a percentage of the larger
/// moving average. Effectively same as Chande's Range Action Verification Index (RAVI).
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn ppo<T: ToPrimitive>(
    data: &[T],
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn apo<T: ToPrimitive>(
    data: &[T],
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, 3).collect::<Vec<f64>>();
///
/// ```
pub fn pmo<T: ToPrimitive>(data: &[T], win1: usize, win2: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(win1).chain(
        smooth::ewma(
            &smooth::ewma(
                &data
                    .windows(2)
                    .map(|pair| {
                        1000.0 * (pair[1].to_f64().unwrap() / pair[0].to_f64().unwrap() - 1.0)
                    })
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2, 4, 8).collect::<Vec<f64>>();
///
/// ```
pub fn ultimate<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
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
        let (h, l, prevc, c) = (
            h.to_f64().unwrap(),
            l.to_f64().unwrap(),
            prevc.to_f64().unwrap(),
            c.to_f64().unwrap(),
        );
        (
            c - f64::min(l, prevc),
            f64::max(h, prevc) - f64::min(l, prevc),
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn pgo<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let tr = _true_range(high, low, close).collect::<Vec<f64>>();
    let atr = iter::once(f64::NAN).chain(smooth::ewma(&tr, window));
    let sma_close = smooth::sma(close, window);
    izip!(close, sma_close, atr)
        .map(|(c, c_ma, tr_ma)| (c.to_f64().unwrap() - c_ma) / tr_ma)
        .collect::<Vec<f64>>()
        .into_iter()
}

pub(crate) fn _swing<'a, T: ToPrimitive>(
    open: &'a [T],
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
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
        let (prevo, o, h, l, prevc, c) = (
            prevo.to_f64().unwrap(),
            o.to_f64().unwrap(),
            h.to_f64().unwrap(),
            l.to_f64().unwrap(),
            prevc.to_f64().unwrap(),
            c.to_f64().unwrap(),
        );

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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     0.5).collect::<Vec<f64>>();
///
/// ```
pub fn si<'a, T: ToPrimitive>(
    open: &'a [T],
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,5.0,2.0],
///     3).collect::<Vec<f64>>();
///
/// ```
pub fn trix<T: ToPrimitive>(close: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn tii<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let win = window.div_ceil(2);
    iter::repeat(f64::NAN).take(win - 1).chain(
        smooth::sma(data, window)
            .zip(data)
            .map(|(avg, actual)| {
                let dev: f64 = actual.to_f64().unwrap() - avg;
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

fn stoch<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
    smooth_win: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let fast_k = smooth::sma(
        &izip!(
            high.windows(window),
            low.windows(window),
            &close[(window - 1)..]
        )
        .map(|(h, l, c)| {
            let hh = h
                .iter()
                .fold(f64::NAN, |state, x| state.max(x.to_f64().unwrap()));
            let ll = l
                .iter()
                .fold(f64::NAN, |state, x| state.min(x.to_f64().unwrap()));
            100.0 * (c.to_f64().unwrap() - ll) / (hh - ll)
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn stochastic<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    stoch(high, low, close, window, 3)
}

fn _stc<T: ToPrimitive>(
    series: &[T],
    window: usize,
    smooth: usize,
) -> impl Iterator<Item = f64> + '_ {
    smooth::wilder(
        &series
            .windows(window)
            .map(|w| {
                let mut hh = f64::NAN;
                let mut ll = f64::NAN;
                for x in w {
                    hh = hh.max(x.to_f64().unwrap());
                    ll = ll.min(x.to_f64().unwrap());
                }
                100.0 * (w[w.len() - 1].to_f64().unwrap() - ll) / (hh - ll)
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,6.0,1.0],
///     2, 3, 5).collect::<Vec<f64>>();
///
/// ```
pub fn stc<T: ToPrimitive>(
    close: &[T],
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn relative_vigor<'a, T: ToPrimitive>(
    open: &'a [T],
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let close_open = open
        .iter()
        .zip(close)
        .map(|(o, c)| c.to_f64().unwrap() - o.to_f64().unwrap())
        .collect::<Vec<f64>>();
    let high_low = high
        .iter()
        .zip(low)
        .map(|(h, l)| h.to_f64().unwrap() - l.to_f64().unwrap())
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn fisher<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let hl2 = high
        .iter()
        .zip(low)
        .map(|(h, l)| (h.to_f64().unwrap() + l.to_f64().unwrap()) / 2.0)
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2, 3).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn rainbow<T: ToPrimitive>(
    data: &[T],
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
                let val = data[x].to_f64().unwrap();
                hlookback = hlookback.max(val);
                llookback = llookback.min(val);
            });
            let osc = 100.0 * (data[i].to_f64().unwrap() - total / 10.0)
                / (hlookback - llookback).max(0.000001);
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     2, 3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn coppock<T: ToPrimitive>(
    data: &[T],
    window: usize,
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(long).chain(
        smooth::wma(
            &(long..data.len())
                .map(|x| {
                    100.0
                        * (data[x].to_f64().unwrap() / data[x - short].to_f64().unwrap()
                            + data[x].to_f64().unwrap() / data[x - long].to_f64().unwrap()
                            - 2.0)
                })
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn roc<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(window - 1).chain(
        data.windows(window)
            .map(|w| 100.0 * (w[w.len() - 1].to_f64().unwrap() / w[0].to_f64().unwrap() - 1.0)),
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<f64>>();
///
/// ```
pub fn bal_power<'a, T: ToPrimitive>(
    open: &'a [T],
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    smooth::ewma(
        &izip!(open, high, low, close)
            .map(|(o, h, l, c)| {
                (c.to_f64().unwrap() - o.to_f64().unwrap())
                    / (h.to_f64().unwrap() - l.to_f64().unwrap())
            })
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn disparity<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.iter()
        .zip(smooth::ewma(data, window))
        .map(|(x, ma)| 100.0 * (x.to_f64().unwrap() - ma) / ma)
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn qstick<'a, T: ToPrimitive>(
    open: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let q = close
        .iter()
        .zip(open.iter())
        .map(|(c, o)| c.to_f64().unwrap() - o.to_f64().unwrap())
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cog<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let weights: Vec<f64> = (1..=window).map(|x| x as f64).collect();
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            -w.iter()
                .rev()
                .zip(weights.iter())
                .map(|(e, i)| e.to_f64().unwrap() * i)
                .sum::<f64>()
                / w.iter().filter_map(|x| x.to_f64()).sum::<f64>()
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn psych<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(window).chain(
        data.windows(2)
            .map(|pair| {
                (pair[1].to_f64().unwrap() - pair[0].to_f64().unwrap())
                    .signum()
                    .max(0.0)
            })
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn tsi<T: ToPrimitive>(
    data: &[T],
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
    let diffs = data
        .windows(2)
        .map(|pair| pair[1].to_f64().unwrap() - pair[0].to_f64().unwrap())
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
    iter::repeat(f64::NAN).take(long).chain(tsi)
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
/// momentum::special_k(&[1.0; 50]).collect::<Vec<f64>>();
///
/// ```
pub fn special_k<T: ToPrimitive>(data: &[T]) -> impl Iterator<Item = f64> + '_ {
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
/// momentum::kst(&[1.0; 50], Some([10, 15, 25, 30]), None).collect::<Vec<f64>>();
///
/// ```
pub fn kst<T: ToPrimitive>(
    data: &[T],
    rocs: Option<[usize; 4]>,
    windows: Option<[usize; 4]>,
) -> impl Iterator<Item = f64> + '_ {
    let rocs = rocs.unwrap_or([10, 15, 20, 30]);
    let periods = windows.unwrap_or([10, 10, 10, 15]);
    const MULTIPLIERS: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    prings(data, &rocs, &periods, &MULTIPLIERS)
}

fn prings<'a, 'b, T: ToPrimitive>(
    data: &'a [T],
    rocs: &'b [usize],
    periods: &'b [usize],
    multipliers: &'b [f64],
) -> impl Iterator<Item = f64> + 'a {
    let mut result = vec![0.0; data.len() - rocs[0] - (periods[0] - 1)];
    izip!(rocs, periods, multipliers).for_each(|(roc_win, ma_win, mult)| {
        let roc_win = roc_win + 1;
        let roc = data
            .windows(roc_win)
            .map(|w| 100.0 * (w[roc_win - 1].to_f64().unwrap() / w[0].to_f64().unwrap() - 1.0))
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
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     2, 3, 2).collect::<Vec<f64>>();
///
/// ```
pub fn derivative<T: ToPrimitive>(
    data: &[T],
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

/// Commodity Channel Index
///
/// Measures the difference between the current price and the historical average price.
///
/// ## Usage
///
/// Value above zero indicates the price is above historical average.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/c/commoditychannelindex.asp)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::cci(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cci<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        izip!(high, low, close)
            .map(|(h, l, c)| {
                (h.to_f64().unwrap() + l.to_f64().unwrap() + c.to_f64().unwrap()) / 3.0
            })
            .collect::<Vec<f64>>()
            .windows(window)
            .map(|w| {
                let avg = w.iter().sum::<f64>() / window as f64;
                let dev = w.iter().fold(0.0, |acc, x| acc + (x - avg).abs()) / window as f64;
                (w.last().unwrap() - avg) / (0.015 * dev)
            })
            .collect::<Vec<f64>>(),
    )
}

/// Quantitative Qualitative Estimation (QQE)
///
/// A derivative of RSI. Effectively a smoothed version to dampen short term changes.
///
/// ## Usage
///
/// When value crosses above signal line, suggests uptrend. Also, a value below 30 or above 70
/// suggests oversold or overbought respectively.
///
/// ## Sources
///
/// [[1]](https://www.lizardindicators.com/the-quantitative-qualitative-estimation-indicator/)
/// [[2]](https://tradingtact.com/qqe-indicator/)
/// [[3]](https://www.prorealcode.com/prorealtime-indicators/qqe-quantitative-qualitative-estimation/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::qqe(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     3, 3).collect::<Vec<f64>>();
///
/// ```
pub fn qqe<T: ToPrimitive>(
    data: &[T],
    window: usize,
    smoothing: usize,
) -> impl Iterator<Item = f64> + '_ {
    let qqe_factor = 4.236;
    let rsi_vals = rsi(data, window).skip(window).collect::<Vec<f64>>();
    let rsi_ma = smooth::ewma(&rsi_vals, smoothing)
        .skip(smoothing - 1)
        .collect::<Vec<f64>>();
    let dar = smooth::wilder(
        &smooth::wilder(
            &rsi_ma
                .windows(2)
                .map(|pair| (pair[1] - pair[0]).abs())
                .collect::<Vec<f64>>(),
            window,
        )
        .skip(window - 1)
        .collect::<Vec<f64>>(),
        window,
    )
    .skip(window - 1)
    .map(|x| x * qqe_factor)
    .collect::<Vec<f64>>();
    let rsi_ma = rsi_ma
        .into_iter()
        .skip(window * 2 - 1)
        .collect::<Vec<f64>>();

    iter::repeat(f64::NAN)
        .take(window * 3 + (smoothing - 1))
        .chain((1..dar.len()).scan(0.0, move |state, x| {
            *state = if rsi_ma[x] > *state {
                if rsi_ma[x - 1] > *state {
                    state.max(rsi_ma[x] - dar[x])
                } else {
                    rsi_ma[x] - dar[x]
                }
            } else if rsi_ma[x] < *state {
                if rsi_ma[x - 1] < *state {
                    state.min(rsi_ma[x] + dar[x])
                } else {
                    rsi_ma[x] + dar[x]
                }
            } else {
                *state
            };
            Some(*state)
        }))
}

/// Detrended Ehlers Leading Indicator
///
/// Computed by subtracting the simple moving average of the detrended synthetic price
/// from the detrended synthetic price.
///
/// ## Usage
///
/// Value above zero suggests an uptrend
///
/// ## Sources
///
/// [[1]](https://www.motivewave.com/studies/detrended_ehlers_leading_indicator.htm)
/// [[2]](https://www.prorealcode.com/prorealtime-indicators/ehlers-detrended-leading-indicator/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::deli(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0,4.0],
///     3).collect::<Vec<f64>>();
///
/// ```
pub fn deli<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let quotes = high
        .windows(2)
        .zip(low.windows(2))
        .map(|(h, l)| {
            (h[0].to_f64().unwrap().max(h[1].to_f64().unwrap())
                + l[0].to_f64().unwrap().min(l[1].to_f64().unwrap()))
                / 2.0
        })
        .collect::<Vec<f64>>();
    let dsp = smooth::ewma(&quotes, window)
        .zip(smooth::ewma(&quotes, 2 * window))
        .map(|(ma1, ma2)| ma1 - ma2)
        .collect::<Vec<f64>>();
    let ma3 = smooth::ewma(&dsp[2 * window - 1..], window);
    iter::repeat(f64::NAN).take(2 * window).chain(
        dsp[2 * window - 1..]
            .iter()
            .zip(ma3)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    )
}

/// Gator Oscillator
///
/// Extends Alligator indicator to create two histograms: delta between Jaws and Teeth, and
/// delta between Teeth and Lips line.
///
/// ## Usage
///
/// Increasing bars on either side of zero line suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://admiralmarkets.com/education/articles/forex-indicators/indicate-market-trend-with-the-gator-oscillator)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::gator(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0],
///     6, 4, 4, 3, 3, 2).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn gator<T: ToPrimitive>(
    data: &[T],
    jaw_win: usize,
    jaw_offset: usize,
    teeth_win: usize,
    teeth_offset: usize,
    lips_win: usize,
    lips_offset: usize,
) -> impl Iterator<Item = (f64, f64)> + '_ {
    alligator(
        data,
        jaw_win,
        jaw_offset,
        teeth_win,
        teeth_offset,
        lips_win,
        lips_offset,
    )
    .map(|x| ((x.0 - x.1).abs(), (x.1 - x.2).abs()))
}

/// KDJ Index
///
/// Indicator consisting of three components: K-line which is a smoothed version of price;
/// D-line which is a smoother version of K-line; and J-line which is a derivative of K and D
/// lines. Effectively stochastic indicator with additional J-line.
///
/// ## Usage
///
/// Suggests an uptrend when J line crosses above K and D lines
///
/// ## Sources
///
/// [[1]](https://market-bulls.com/kdj-indicator/)
/// [[2]](https://www.liberatedstocktrader.com/kdj-indicator/)
///
/// # Examples
///
/// ```
/// use traquer::momentum;
///
/// momentum::kdj(
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &[1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 2, None, Some(1.0)).collect::<Vec<(f64, f64, f64)>>();
///
/// ```
pub fn kdj<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    k_win: usize,
    d_win: usize,
    k_factor: Option<f64>,
    d_factor: Option<f64>,
) -> impl Iterator<Item = (f64, f64, f64)> + 'a {
    let k_factor = k_factor.unwrap_or(3.0);
    let d_factor = d_factor.unwrap_or(2.0);
    stoch(high, low, close, k_win, d_win).map(move |(k, d)| (k, d, k_factor * k - d_factor * d))
}
