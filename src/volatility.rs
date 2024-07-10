//! Volatility Indicators
//!
//! Indicators that measure the price movement, regardless of direction. In essence, it is
//! signaling whether there is a trend or not generally based on the delta between the
//! highest and lowest prices in period. It may also be represented as channels for which
//! it expects prices to fall within.

use std::iter;

use itertools::izip;

use crate::smooth;

/// Mass Index
///
/// Measures the volatility of stock prices by calculating a ratio of two exponential
/// moving averages of the high-low differential.
///
/// ## Usage
///
/// A high value suggests a potential reversal in trend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/m/mass-index.asp)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::mass(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn mass<'a>(
    high: &'a [f64],
    low: &'a [f64],
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + 'a {
    let ma1 = smooth::ewma(
        &high
            .iter()
            .zip(low)
            .map(|(h, l)| (h - l))
            .collect::<Vec<f64>>(),
        short,
    )
    .collect::<Vec<f64>>();
    let ma2 = smooth::ewma(&ma1[short - 1..], short);
    iter::repeat(f64::NAN).take((long - 1) + (short - 1)).chain(
        ma1.iter()
            .skip(short - 1)
            .zip(ma2)
            .map(|(num, denom)| num / denom)
            .collect::<Vec<f64>>()
            .windows(long)
            .map(|w| w.iter().sum::<f64>())
            .collect::<Vec<f64>>(),
    )
}

/// Keltner Channel
///
/// Upper and lower bands are defined by True Range from moving average.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/k/keltnerchannel.asp)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::keltner(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<(f64,f64,f64)>>();
///
/// ```
pub fn keltner<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64, f64)> + 'a {
    smooth::ewma(close, window)
        .zip(atr(high, low, close, window))
        .map(|(middle, atr)| (middle, middle + 2.0 * atr, middle - 2.0 * atr))
}

/// Gopalakrishnan Range Index
///
/// Calculates the range of a security's price action over a specified period,
/// providing insights into the volatility
///
/// ## Usage
///
/// A high value suggests a strong trend while a low value suggests sideways movement.
///
/// ## Sources
///
/// [[1]](https://superalgos.org/Docs/Foundations/Topic/gapo.shtml)
/// [[2]](https://library.tradingtechnologies.com/trade/chrt-ti-gopalakrishnan-range-index.html)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::gri(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn gri<'a>(high: &'a [f64], low: &'a [f64], window: usize) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        high.windows(window)
            .zip(low.windows(window))
            .map(move |(h, l)| {
                let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
                let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
                f64::ln(hh - ll) / f64::ln(window as f64)
            }),
    )
}

pub(crate) fn _true_range<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
) -> impl Iterator<Item = f64> + 'a {
    izip!(&high[1..], &low[1..], &close[..close.len() - 1])
        .map(|(h, l, prevc)| (h - l).max(f64::abs(h - prevc)).max(f64::abs(l - prevc)))
}

/// True Range
///
/// Measures market volatility by computing the greatest of the following: current high less
/// the current low; the absolute value of the current high less the previous close;
/// and the absolute value of the current low less the previous close.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/a/atr.asp)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::tr(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     ).collect::<Vec<f64>>();
///
/// ```
pub fn tr<'a>(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(_true_range(high, low, close))
}

/// Average True Range
///
/// Moving average of the True Range series
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/a/atr.asp)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::atr(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn atr<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(
        smooth::wilder(&_true_range(high, low, close).collect::<Vec<f64>>(), window)
            .collect::<Vec<f64>>(),
    )
}

/// Typical Price
///
/// Average of a given day's high, low, and close price.
///
/// ## Sources
///
/// [[1]](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/typical-price)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::typical(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn typical<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    smooth::sma(
        &izip!(high, low, close)
            .map(|(h, l, c)| (h + l + c) / 3.0)
            .collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Standard Deviation
///
/// Measures market volatility. Standard deviation of price over a period.
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::std_dev(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, Some(1.0)).collect::<Vec<f64>>();
///
/// ```
pub fn std_dev(
    data: &[f64],
    window: usize,
    deviations: Option<f64>,
) -> impl Iterator<Item = f64> + '_ {
    let devs = deviations.unwrap_or(2.0);
    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(smooth::std_dev(data, window).map(move |x| x * devs))
}

/// Bollinger Bands
///
/// Channels defined by standard deviations away from a moving average.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/b/bollingerbands.asp)
///
/// # Examples
///
/// ```
/// use traquer::{volatility, smooth};
///
/// volatility::bbands(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, Some(1.0), Some(smooth::MaMode::SMA)).collect::<Vec<(f64,f64,f64)>>();
///
/// ```
pub fn bbands(
    data: &[f64],
    window: usize,
    deviations: Option<f64>,
    mamode: Option<smooth::MaMode>,
) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
    smooth::ma(data, window, mamode.unwrap_or(smooth::MaMode::EWMA))
        .zip(std_dev(data, window, deviations))
        .map(|(ma, stdev)| (ma + stdev, ma, ma - stdev))
}

/// Donchian Channels
///
/// Channels defined by highest high and lowest low
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::donchian(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<(f64,f64,f64)>>();
///
/// ```
pub fn donchian<'a>(
    high: &'a [f64],
    low: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64, f64)> + 'a {
    iter::repeat((f64::NAN, f64::NAN, f64::NAN))
        .take(window - 1)
        .chain(high.windows(window).zip(low.windows(window)).map(|(h, l)| {
            let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
            let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
            (hh, (hh + ll) / 2.0, ll)
        }))
}

/// Fractal Chaos Bands
///
/// Channels defined by peaks or valleys in prior period.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/script/Yy2ASjTq-Fractal-Chaos-Bands/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::fbands(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     ).collect::<Vec<(f64,f64)>>();
///
/// ```
pub fn fbands<'a>(high: &'a [f64], low: &'a [f64]) -> impl Iterator<Item = (f64, f64)> + 'a {
    let win = 5;
    iter::repeat((f64::NAN, f64::NAN)).take(win - 1).chain(
        high.windows(win)
            .zip(low.windows(win))
            .scan((0.0, 0.0), |state, (h, l)| {
                let (hh, _) = h.iter().enumerate().fold((0, h[0]), |state, (idx, x)| {
                    if x > &state.1 {
                        (idx, *x)
                    } else {
                        state
                    }
                });
                let upper = if hh == 2 { h[2] } else { state.0 };

                let (ll, _) = l.iter().enumerate().fold((0, l[0]), |state, (idx, x)| {
                    if x < &state.1 {
                        (idx, *x)
                    } else {
                        state
                    }
                });
                let lower = if ll == 2 { l[2] } else { state.1 };
                *state = (upper, lower);
                Some(*state)
            }),
    )
}

/// Historical Volatility
///
/// Measures the standard deviation of returns, annualised.
///
/// ## Usage
///
/// A higher value suggests higher risk.
///
/// ## Sources
///
/// [[1]](https://www.macroption.com/historical-volatility-calculation/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::hv(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, Some(1.0)).collect::<Vec<f64>>();
///
/// ```
pub fn hv(data: &[f64], window: usize, deviations: Option<f64>) -> impl Iterator<Item = f64> + '_ {
    let annualize = 252.0 * deviations.unwrap_or(1.0);
    iter::repeat(f64::NAN).take(window).chain(
        data.windows(2)
            .map(|pair| f64::ln(pair[1] / pair[0]))
            .collect::<Vec<f64>>()
            .windows(window)
            .map(move |w| {
                let mean = w.iter().sum::<f64>() / window as f64;
                (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() * annualize / window as f64)
                    .sqrt()
            })
            .collect::<Vec<f64>>(),
    )
}

/// Stoller Average Range Channel (STARC)
///
/// Upper and lower bands are defined by True Range from moving average with a multiplier.
/// Similar to Keltner Channels.
///
/// [[1]](https://www.investopedia.com/terms/s/starc.asp)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::starc(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, 2, Some(1.3)).collect::<Vec<(f64,f64)>>();
///
/// ```
pub fn starc<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
    ma_window: usize,
    multiplier: Option<f64>,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let multiplier = multiplier.unwrap_or(1.2);
    atr(high, low, close, window)
        .zip(smooth::sma(close, ma_window))
        .map(move |(atr, ma)| (ma + multiplier * atr, ma - multiplier * atr))
}

/// Chaikin volatility
///
/// Measures the volatility of a security's price action by comparing the spread between
/// the high and low prices over a specified period.
///
/// ## Usage
///
/// A high value suggests high volatility.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/chart/AUDUSD/gjfxqWqW-What-Is-a-Chaikin-Volatility-Indicator-in-Trading/)
/// [[2]](https://theforexgeek.com/chaikins-volatility-indicator/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::cvi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, 3).collect::<Vec<f64>>();
///
/// ```
pub fn cvi<'a>(
    high: &'a [f64],
    low: &'a [f64],
    window: usize,
    rate_of_change: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(rate_of_change).chain(
        smooth::ewma(
            &high
                .iter()
                .zip(low)
                .map(|(h, l)| h - l)
                .collect::<Vec<f64>>(),
            window,
        )
        .collect::<Vec<f64>>()
        .windows(rate_of_change + 1)
        .map(|w| 100.0 * (w.last().unwrap() / w.first().unwrap() - 1.0))
        .collect::<Vec<f64>>(),
    )
}

/// Relative Volatility
///
/// Measures the direction and magnitude of volatility in an asset’s price. Unlike the
/// Relative Strength Index (RSI), which uses absolute prices, the RVI uses standard deviation.
///
/// ## Usage
///
/// A high value suggests higher volatility.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/support/solutions/43000594684-relative-volatility-index/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::relative_vol(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, 2).collect::<Vec<f64>>();
///
/// ```
pub fn relative_vol(
    close: &[f64],
    window: usize,
    smoothing: usize,
) -> impl Iterator<Item = f64> + '_ {
    let (gain, loss): (Vec<f64>, Vec<f64>) = izip!(
        smooth::std_dev(close, window),
        &close[window - 1..],
        &close[window - 2..close.len() - 1]
    )
    .map(|(std, curr, prev)| {
        (
            f64::max(0.0, (curr - prev).signum()) * std,
            f64::max(0.0, (prev - curr).signum()) * std,
        )
    })
    .unzip();
    iter::repeat(f64::NAN).take(window - 1).chain(
        smooth::wilder(&gain, smoothing)
            .zip(smooth::wilder(&loss, smoothing))
            .map(|(g, l)| 100.0 * g / (g + l))
            .collect::<Vec<f64>>(),
    )
}

/// Inertia
///
/// An extension of Donald Dorsey’s Relative Volatility Index (RVI). The name “Inertia”
/// reflects the concept that trends require more force to reverse than to continue
/// in the same direction.
///
/// ## Usage
///
/// A high value suggests higher volatility.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/script/ODEBlQkx-Inertia-Indicator/)
/// [[2]](https://theforexgeek.com/inertia-indicator/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::inertia(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 2).collect::<Vec<f64>>();
///
/// ```
pub fn inertia(close: &[f64], window: usize, smoothing: usize) -> impl Iterator<Item = f64> + '_ {
    smooth::lrf(
        &relative_vol(close, window, window).collect::<Vec<f64>>(),
        smoothing,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Choppiness Index
///
/// Aims to capture the true volatility and directionality of the market by taking
/// into account the range between the highest high and the lowest low prices over
/// a specified period.
///
/// ## Usage
///
/// A high value suggests a strong trend while a low value suggests sideways movement.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/support/solutions/43000501980-choppiness-index-chop/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::chop(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn chop<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window).chain(
        izip!(
            high[1..].windows(window),
            low[1..].windows(window),
            _true_range(high, low, close)
                .collect::<Vec<f64>>()
                .windows(window)
        )
        .map(|(h, l, tr)| {
            let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
            let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
            let tr_sum = tr.iter().sum::<f64>();
            100.0 * f64::ln(tr_sum / (hh - ll)) / f64::ln(window as f64)
        })
        .collect::<Vec<f64>>(),
    )
}

/// Vertical Horizontal Filter
///
/// Measures the level of trend activity in a financial market by comparing the max price
/// range over a specific period to the cumulative price movement within that period.
///
/// ## Usage
///
/// A higher VHF value indicates a trending market, while lower VHF suggests consolidation.
///
/// ## Sources
///
/// [[1]](https://www.upcomingtrader.com/blog/the-vertical-horizontal-filter-a-traders-guide-to-market-phases/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::vhf(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn vhf<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let diffs = close
        .windows(2)
        .map(|pair| {
            let (prev, curr) = (pair[0], pair[1]);
            (curr - prev).abs()
        })
        .collect::<Vec<f64>>();
    iter::repeat(f64::NAN).take(window).chain(
        izip!(
            diffs.windows(window),
            high.windows(window).skip(1),
            low.windows(window).skip(1)
        )
        .map(|(diff, h, l)| {
            (h.iter().fold(f64::NAN, |state, &x| state.max(x))
                - l.iter().fold(f64::NAN, |state, &x| state.min(x)))
                / diff.iter().sum::<f64>()
        })
        .collect::<Vec<f64>>(),
    )
}

/// Heikin-Ashi Candlestick
///
/// Uses a modified formula based on two-period averages to derive OHLC candlesticks
///
/// ## Usage
///
/// Like normal chandlestick chart.
///
/// ## Sources
///
/// [[1]](https://www.marketvolume.com/technicalanalysis/heikin_ashi_candlesticks.asp)
/// [[2]](https://www.investopedia.com/terms/h/heikinashi.asp)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::heikin_ashi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     ).collect::<Vec<(f64,f64,f64,f64)>>();
///
/// ```
pub fn heikin_ashi<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
) -> impl Iterator<Item = (f64, f64, f64, f64)> + 'a {
    let c_0 = (open[0] + high[0] + low[0] + close[0]) / 4.0;
    let o_0 = (open[0] + close[0]) / 2.0;
    iter::once((o_0, high[0], low[0], c_0)).chain(
        izip!(
            open[1..].iter(),
            high[1..].iter(),
            low[1..].iter(),
            close[1..].iter()
        )
        .scan((o_0, c_0), |(prevo, prevc), (o, h, l, c)| {
            let c_i = (o + h + l + c) / 4.0;
            let o_i = (*prevo + *prevc) / 2.0;
            let h_i = h.max(o_i).max(c_i);
            let l_i = l.min(o_i).min(c_i);
            *prevo = o_i;
            *prevc = c_i;
            Some((o_i, h_i, l_i, c_i))
        }),
    )
}

/// Weighted Closing Price
///
/// Doubles the closing price.
///
/// ## Usage
///
/// Similar to ATR.
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::wcp(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     ).collect::<Vec<f64>>();
///
/// ```
pub fn wcp<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
) -> impl Iterator<Item = f64> + 'a {
    izip!(high, low, close).map(|(h, l, c)| (h + l + 2.0 * c) / 4.0)
}

/// Close-to-Close Historical Volatility
///
/// The simplest volatility estimator. Calculated using only stock's closing prices. Value is annualised.
///
/// ## Usage
///
/// Higher value implies increase volatility.
///
/// ## Sources
///
/// [[1]](https://harbourfronts.com/close-close-historical-volatility-calculation-volatility-analysis-python/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::cc_hv(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<f64>>();
///
/// ```
pub fn cc_hv(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    iter::repeat(f64::NAN).take(window).chain(
        data.windows(2)
            .map(|w| (w[1] / w[0]).ln().powi(2))
            .collect::<Vec<f64>>()
            .windows(window)
            .map(|w| (w.iter().sum::<f64>() * 252.0 / window as f64).sqrt())
            .collect::<Vec<f64>>(),
    )
}

/// Garman-Klass-Yang-Zhang Historical Volatility
///
/// Extends Garman-Klass volatility estimator by taking into consideration overnight jumps. Value
/// is annualised.
///
/// ## Usage
///
/// Higher value implies increase volatility.
///
/// ## Sources
///
/// [[1]](https://harbourfronts.com/garman-klass-yang-zhang-historical-volatility-calculation-volatility-analysis-python/)
///
/// # Examples
///
/// ```
/// use traquer::volatility;
///
/// volatility::gkyz_hv(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<f64>>();
///
/// ```
pub fn gkyz_hv<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window).chain(
        izip!(&open[1..], &high[1..], &low[1..], &close[1..], close)
            .map(|(o, h, l, c, c0)| {
                (o / c0).ln().powi(2) + 0.5 * (h / l).ln().powi(2)
                    - (2.0 * 2_f64.ln() - 1.0) * (c / o).ln().powi(2)
            })
            .collect::<Vec<f64>>()
            .windows(window)
            .map(|w| (w.iter().sum::<f64>() * 252.0 / window as f64).sqrt())
            .collect::<Vec<f64>>(),
    )
}
