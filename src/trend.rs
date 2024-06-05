//! Trend Indicators
//!
//! Indicators where the direction may signify opportunities. The slope and trajectory of the
//! indicator are more important than the actual value.

use std::iter;

use itertools::{izip, multiunzip};

use crate::momentum::_swing;
use crate::smooth;

/// Quick stick
///
/// Measures buying and selling pressure, taking an average of the difference between
/// closing and opening prices. When the price is closing lower than it opens, the
/// indicator moves lower. When the price is closing higher than the open,
/// the indicator moves up
///
/// # Source
///
/// https://www.investopedia.com/terms/q/qstick.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::qstick(
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

/// Shinohara intensity ratio
///
/// Measures trend intensity by plotting Strong Ratio (Strength) and Weak Ratio (Popularity) lines.
/// The Strong Ratio is (high - prev close) / (prev close - low) and the weak ratio
/// is (high - close) / (close - low).
///
/// NOTE: Implementation differs from source where weak ratio uses close rather than open.
///
/// # Source
///
/// https://www.sevendata.co.jp/shihyou/technical/shinohara.html
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::shinohara(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<(f64,f64)>>();
///
/// ```
pub fn shinohara<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let high_win = high
        .windows(window)
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    let low_win = low
        .windows(window)
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    let close_win = close
        .windows(window)
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    // yahoo uses close rather than open for weak ratio described above
    let weak_ratio = izip!(&high_win, &low_win, &close_win)
        .map(|(h, l, c)| 100.0 * (h - c) / (c - l))
        .collect::<Vec<f64>>();
    let strong_ratio = izip!(
        &high_win[1..],
        &low_win[1..],
        &close_win[..close_win.len() - 1]
    )
    .map(|(h, l, c)| 100.0 * (h - c) / (c - l))
    .collect::<Vec<f64>>();
    iter::repeat(f64::NAN)
        .take(1)
        .chain(strong_ratio)
        .zip(weak_ratio)
}

/// Average directional index
///
/// Measures strength a trend, not the direction, by directional movement by comparing
/// the difference between two consecutive lows with the difference between their
/// respective highs.
///
/// # Source
///
/// https://www.investopedia.com/terms/a/adx.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::adx(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<(f64,f64,f64)>>();
///
/// ```
pub fn adx<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
    smoothing: usize,
) -> impl Iterator<Item = (f64, f64, f64)> + 'a {
    let (dm_pos, dm_neg, tr): (Vec<_>, Vec<_>, Vec<_>) = multiunzip(
        izip!(
            &high[..high.len() - 1],
            &high[1..],
            &low[..low.len() - 1],
            &low[1..],
            &close[..close.len() - 1],
        )
        .map(|(prevh, h, prevl, l, prevc)| {
            let dm_pos = if h - prevh > prevl - l {
                f64::max(0.0, h - prevh)
            } else {
                0.0
            };
            let dm_neg = if prevl - l > h - prevh {
                f64::max(0.0, prevl - l)
            } else {
                0.0
            };
            let tr = (h - l).max(f64::abs(h - prevc)).max(f64::abs(l - prevc));
            (dm_pos, dm_neg, tr)
        }),
    );
    let atr = smooth::wilder(&tr, window).collect::<Vec<f64>>();
    let di_pos = izip!(smooth::wilder(&dm_pos, window), &atr)
        .map(|(di, tr)| di / tr * 100.0)
        .collect::<Vec<f64>>();
    let di_neg = izip!(smooth::wilder(&dm_neg, window), &atr)
        .map(|(di, tr)| di / tr * 100.0)
        .collect::<Vec<f64>>();
    let dx = izip!(&di_pos, &di_neg)
        .map(|(pos, neg)| f64::abs(pos - neg) / (pos + neg) * 100.0)
        .collect::<Vec<f64>>();
    izip!(
        di_pos,
        di_neg,
        iter::repeat(f64::NAN)
            .take(smoothing - 1)
            .chain(smooth::wilder(&dx, smoothing).collect::<Vec<f64>>()),
    )
}

/// Centre of gravity
///
/// Calculates the midpoint of a security's price action over a specified period. Used in
/// tandem with a signal line.
///
/// # Source
///
/// https://www.stockmaniacs.net/center-of-gravity-indicator/
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::cog(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cog(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let weights: Vec<f64> = (1..=window).map(|x| x as f64).collect();
    data.windows(window).map(move |w| {
        -w.iter()
            .rev()
            .zip(weights.iter())
            .map(|(e, i)| e * i)
            .sum::<f64>()
            / w.iter().sum::<f64>()
    })
}

/// Vortex
///
/// Calculates two lines: VI+ and VI-. The greater the distance between the low of a price bar and
/// the subsequent bar's high, the greater the positive Vortex movement (VM+).
///
/// # Source
///
/// https://en.wikipedia.org/wiki/Vortex_indicator
/// https://www.investopedia.com/terms/v/vortex-indicator-vi.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::vortex(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<(f64,f64)>>();
///
/// ```
pub fn vortex<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    izip!(
        &high[..high.len() - 1],
        &high[1..],
        &low[..low.len() - 1],
        &low[1..],
        &close[..close.len() - 1],
    )
    .map(|(prevh, h, prevl, l, prevc)| {
        let vm_pos = (h - prevl).abs();
        let vm_neg = (l - prevh).abs();
        let tr = (h - l).max(f64::abs(h - prevc)).max(f64::abs(l - prevc));
        (vm_pos, vm_neg, tr)
    })
    .collect::<Vec<(f64, f64, f64)>>()
    .windows(window)
    .map(|w| {
        let (vm_pos, vm_neg, tr) = w
            .iter()
            .copied()
            .reduce(|(acc_pos, acc_neg, acc_tr), (pos, neg, tr)| {
                (acc_pos + pos, acc_neg + neg, acc_tr + tr)
            })
            .unwrap();
        (vm_pos / tr, vm_neg / tr)
    })
    .collect::<Vec<(f64, f64)>>()
    .into_iter()
}

/// Vertical horizontal filter
///
/// Measures the level of trend activity in a financial market by comparing the max price
/// range over a specific period to the cumulative price movement within that period.
///
/// # Source
///
/// https://www.upcomingtrader.com/blog/the-vertical-horizontal-filter-a-traders-guide-to-market-phases/
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::vhf(
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
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Accumulative Swing Index
///
/// Cumulative sum of Swing Index
///
/// # Source
///
/// https://www.investopedia.com/terms/a/asi.asp
/// https://quantstrategy.io/blog/accumulative-swing-index-how-to-trade/
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::asi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     0.5).collect::<Vec<f64>>();
///
/// ```
pub fn asi<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    limit: f64,
) -> impl Iterator<Item = f64> + 'a {
    _swing(open, high, low, close, limit).scan(0.0, |acc, x| {
        *acc += x;
        Some(*acc)
    })
}

/// Ulcer Index
///
/// Measures downside risk in terms of both the depth and duration of price declines.
/// The index increases in value as the price moves farther away from a recent high
/// and falls as the price rises to new highs.
///
/// # Source
///
/// https://en.wikipedia.org/wiki/Ulcer_index
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::ulcer(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn ulcer(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let highest = data
        .windows(window)
        .map(|w| w.iter().fold(f64::NAN, |state, &x| state.max(x)));
    smooth::sma(
        &highest
            .zip(data.iter().skip(window - 1))
            .map(|(high, c)| (100.0 * (c - high) / high).powi(2))
            .collect::<Vec<f64>>(),
        window,
    )
    .map(|x| x.sqrt())
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Supertrend
///
/// Acts as a dynamic level of support or resistance.
///
/// # Source
///
/// https://www.tradingview.com/support/solutions/43000634738-supertrend/
/// https://www.investopedia.com/supertrend-indicator-7976167
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::supertrend(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, 3.0).collect::<Vec<f64>>();
///
/// ```
pub fn supertrend<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
    multiplier: f64,
) -> impl Iterator<Item = f64> + 'a {
    // TODO: needs a test for when it actually flips to use upper band line
    let tr = _true_range(high, low, close).collect::<Vec<f64>>();
    let atr = smooth::wilder(&tr, window);
    izip!(&high[window..], &low[window..], &close[window..], atr)
        .scan(
            (f64::NAN, f64::NAN, f64::MIN_POSITIVE, 1),
            |state, (h, l, c, tr)| {
                let (prevlower, prevupper, prevc, prevdir) = state;
                let mut lower = (h + l) / 2.0 - multiplier * tr;
                let mut upper = (h + l) / 2.0 + multiplier * tr;
                if prevc > prevlower && *prevlower > lower {
                    lower = *prevlower;
                }
                if prevc < prevupper && *prevupper < upper {
                    upper = *prevupper;
                }
                let dir = if c > prevupper {
                    1
                } else if c < prevlower {
                    -1
                } else {
                    *prevdir
                };
                *state = (lower, upper, *c, dir);
                if dir > 0 {
                    Some(lower)
                } else {
                    Some(upper)
                }
            },
        )
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Random Walk Index
///
/// Compares a security's price movements to random movements to determine if it's in a trend.
///
/// NOTE: Window includes current price where other libraries use window strictly as lookback.
/// You may need to add 1 to window for comparable behaviour.
///
/// # Source
///
/// https://www.technicalindicators.net/indicators-technical-analysis/168-rwi-random-walk-index
/// https://www.investopedia.com/terms/r/random-walk-index.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::rwi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn rwi<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    // looks back n number of periods *including* current. other libs may not include current.
    izip!(
        high[1..].windows(window),
        low[1..].windows(window),
        _true_range(high, low, close)
            .collect::<Vec<f64>>()
            .windows(window),
    )
    .map(|(h, l, tr)| {
        let mut rwi_high: f64 = 0.0;
        let mut rwi_low: f64 = 0.0;
        let mut tr_sum = 0.0;
        for i in 2..=window {
            tr_sum += tr[window - i];
            let denom = (tr_sum / (i - 1) as f64) * ((i - 1) as f64).sqrt();
            rwi_high = rwi_high.max((h[window - 1] - l[window - i]) / denom);
            rwi_low = rwi_low.max((h[window - i] - l[window - 1]) / denom);
        }
        (rwi_high, rwi_low)
    })
    .collect::<Vec<(f64, f64)>>()
    .into_iter()
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
/// # Source
///
/// https://tradingliteracy.com/psychological-line-indicator/
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::psych(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn psych(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.windows(2)
        .map(|pair| (pair[1] - pair[0]).signum().max(0.0))
        .collect::<Vec<f64>>()
        .windows(window)
        .map(|w| w.iter().sum::<f64>() * 100.0 / window as f64)
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Mass Index
///
/// Measures the volatility of stock prices by calculating a ratio of two exponential
/// moving averages of the high-low differential.
///
/// # Source
///
/// https://www.investopedia.com/terms/m/mass-index.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::mass(
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
    let ma2 = smooth::ewma(&ma1, short);
    ma1.iter()
        .skip(short - 1)
        .zip(ma2)
        .map(|(num, denom)| num / denom)
        .collect::<Vec<f64>>()
        .windows(long)
        .map(|w| w.iter().sum::<f64>())
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Keltner Channel
///
/// Upper and lower bands are defined by True Range from moving average.
///
/// # Source
///
/// https://www.investopedia.com/terms/k/keltnerchannel.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::keltner(
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
        .zip(iter::once(f64::NAN).chain(atr(high, low, close, window)))
        .map(|(middle, atr)| (middle, middle + 2.0 * atr, middle - 2.0 * atr))
}

/// Gopalakrishnan Range Index
///
/// Calculates the range of a security's price action over a specified period,
/// providing insights into the volatility
///
/// # Source
///
/// https://library.tradingtechnologies.com/trade/chrt-ti-gopalakrishnan-range-index.html
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::gri(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn gri<'a>(high: &'a [f64], low: &'a [f64], window: usize) -> impl Iterator<Item = f64> + 'a {
    high.windows(window)
        .zip(low.windows(window))
        .map(move |(h, l)| {
            let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
            let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
            f64::ln(hh - ll) / f64::ln(window as f64)
        })
}

pub(crate) fn _true_range<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
) -> impl Iterator<Item = f64> + 'a {
    izip!(&high[1..], &low[1..], &close[..close.len() - 1])
        .map(|(h, l, prevc)| (h - l).max(f64::abs(h - prevc)).max(f64::abs(l - prevc)))
}

/// True range
///
/// Measures market volatility by computing the greatest of the following: current high less
/// the current low; the absolute value of the current high less the previous close;
/// and the absolute value of the current low less the previous close.
///
/// # Source
///
/// https://www.investopedia.com/terms/a/atr.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::tr(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     ).collect::<Vec<f64>>();
///
/// ```
pub fn tr<'a>(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    _true_range(high, low, close)
}

/// Average true range
///
/// Moving average of the True Range series
///
/// https://www.investopedia.com/terms/a/atr.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::atr(
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
    smooth::wilder(&_true_range(high, low, close).collect::<Vec<f64>>(), window)
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Typical price
///
/// Average of a given day's high, low, and close price.
///
/// # Source
///
/// https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/typical-price
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::typical(
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
/// use traquer::trend;
///
/// trend::std_dev(
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
    smooth::std_dev(data, window).map(move |x| x * devs)
}

/// Bollinger Bands
///
/// Channels defined by standard deviations away from a moving average.
///
/// # Source
///
/// https://www.investopedia.com/terms/b/bollingerbands.asp
///
/// # Examples
///
/// ```
/// use traquer::{trend, smooth};
///
/// trend::bbands(
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
/// # Source
///
/// https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::donchian(
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
    high.windows(window).zip(low.windows(window)).map(|(h, l)| {
        let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
        let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
        (hh, (hh + ll) / 2.0, ll)
    })
}

/// Fractal Chaos Bands
///
/// Channels defined by peaks or valleys in prior period.
///
/// # Source
///
/// https://www.tradingview.com/script/Yy2ASjTq-Fractal-Chaos-Bands/
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::fbands(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     ).collect::<Vec<(f64,f64)>>();
///
/// ```
pub fn fbands<'a>(high: &'a [f64], low: &'a [f64]) -> impl Iterator<Item = (f64, f64)> + 'a {
    high.windows(5)
        .zip(low.windows(5))
        .scan((0.0, 0.0), |state, (h, l)| {
            let (hh, _) =
                h.iter().enumerate().fold(
                    (0, h[0]),
                    |state, (idx, x)| {
                        if x > &state.1 {
                            (idx, *x)
                        } else {
                            state
                        }
                    },
                );
            let upper = if hh == 2 { h[2] } else { state.0 };

            let (ll, _) =
                l.iter().enumerate().fold(
                    (0, l[0]),
                    |state, (idx, x)| {
                        if x < &state.1 {
                            (idx, *x)
                        } else {
                            state
                        }
                    },
                );
            let lower = if ll == 2 { l[2] } else { state.1 };
            *state = (upper, lower);
            Some(*state)
        })
}

/// Historical Volatility
///
/// Measures the standard deviation of returns, annualised.
///
/// # Source
///
/// https://www.macroption.com/historical-volatility-calculation/
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::hv(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, Some(1.0)).collect::<Vec<f64>>();
///
/// ```
pub fn hv(data: &[f64], window: usize, deviations: Option<f64>) -> impl Iterator<Item = f64> + '_ {
    let annualize = 252.0 * deviations.unwrap_or(1.0);
    data.windows(2)
        .map(|pair| f64::ln(pair[1] / pair[0]))
        .collect::<Vec<f64>>()
        .windows(window)
        .map(move |w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() * annualize / window as f64).sqrt()
        })
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Stoller Average Range Channel (STARC)
///
/// Upper and lower bands are defined by True Range from moving average with a multiplier.
/// Similar to Keltner Channels.
///
/// https://www.investopedia.com/terms/s/starc.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::starc(
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
        .skip(std::cmp::max(0_i32, (ma_window - 1) as i32 - window as i32) as usize)
        .zip(
            smooth::sma(close, ma_window)
                .skip(std::cmp::max(0_i32, window as i32 - (ma_window - 1) as i32) as usize),
        )
        .map(move |(atr, ma)| (ma + multiplier * atr, ma - multiplier * atr))
}

/// Parabolic Stop and Reverse (SAR)
///
/// Calculating the stop for each upcoming period. When the stop is hit you close the
/// current trade and initiate a new trade in the opposite direction.
///
/// # Source
///
/// https://www.investopedia.com/terms/p/parabolicindicator.asp
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::psar(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     None, None).collect::<Vec<f64>>();
///
/// ```
pub fn psar<'a>(
    high: &'a [f64],
    low: &'a [f64],
    af: Option<f64>,
    af_max: Option<f64>,
) -> impl Iterator<Item = f64> + 'a {
    let af_inc = af.unwrap_or(0.02);
    let mut af = af_inc;
    let af_max = af_max.unwrap_or(0.2);
    let mut fall = true;
    let mut sar = high[0];
    let mut ep = low[0];

    let mut flip: bool;
    let mut result = Vec::with_capacity(high.len() - 1);
    for i in 0..high.len() - 1 {
        sar = sar + af * (ep - sar);
        if fall {
            flip = high[i + 1] > sar;
            if low[i + 1] < ep {
                ep = low[i + 1];
                af = f64::min(af + af_inc, af_max);
            }
            sar = f64::max(f64::max(high[std::cmp::max(i, 1) - 1], high[i + 1]), sar)
        } else {
            flip = low[i + 1] < sar;
            if high[i + 1] > ep {
                ep = high[i + 1];
                af = f64::min(af + af_inc, af_max);
            }
            sar = f64::min(f64::min(low[std::cmp::max(i, 1) - 1], low[i + 1]), sar)
        }

        if flip {
            sar = ep;
            af = af_inc;
            fall = !fall;
            ep = if fall { low[i + 1] } else { high[i + 1] };
        }
        result.push(sar);
    }
    result.into_iter()
}
