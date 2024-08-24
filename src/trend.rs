//! Trend Indicators
//!
//! Indicators where the direction may signify opportunities. The slope and trajectory of the
//! indicator are more important than the magnitude of the resulting value.
//! In some cases, the strength of the trend can be derived.

use std::iter;

use itertools::{izip, multiunzip};
use num_traits::cast::ToPrimitive;

use crate::momentum::_swing;
use crate::smooth;
use crate::volatility::{_true_range, atr};

/// Shinohara Intensity Ratio
///
/// Measures trend intensity by plotting Strong Ratio (Strength) and Weak Ratio (Popularity) lines.
/// The Strong Ratio is (high - prev close) / (prev close - low) and the Weak Ratio
/// is (high - close) / (close - low).
///
/// ## Usage
///
/// When Strong Ratio is above the Weak, it suggests uptrend with greater intensity the greater,
/// the delta.
///
/// NOTE: Implementation differs from source where weak ratio uses close rather than open.
///
/// ## Sources
///
/// [[1]](https://www.sevendata.co.jp/shihyou/technical/shinohara.html)
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
pub fn shinohara<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let high_win = high
        .windows(window)
        .map(|w| w.iter().filter_map(|x| x.to_f64()).sum())
        .collect::<Vec<f64>>();
    let low_win = low
        .windows(window)
        .map(|w| w.iter().filter_map(|x| x.to_f64()).sum())
        .collect::<Vec<f64>>();
    let close_win = close
        .windows(window)
        .map(|w| w.iter().filter_map(|x| x.to_f64()).sum())
        .collect::<Vec<f64>>();
    // NOTE: using close rather than open for weak ratio described above
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
        .take(window)
        .chain(strong_ratio)
        .zip(iter::repeat(f64::NAN).take(window - 1).chain(weak_ratio))
}

/// Average Directional Index (ADX)
///
/// Measures strength a trend, not the direction, by directional movement by comparing
/// the difference between two consecutive lows with the difference between their
/// respective highs.
///
/// ## Usage
///
/// When +DMI is above -DMI, it suggests an uptrend. A higher ADX value suggest strength of trend
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/a/adx.asp)
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
pub fn adx<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
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
            let (prevh, h, prevl, l, prevc) = (
                prevh.to_f64().unwrap(),
                h.to_f64().unwrap(),
                prevl.to_f64().unwrap(),
                l.to_f64().unwrap(),
                prevc.to_f64().unwrap(),
            );
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
    let dx = izip!(&di_pos[window - 1..], &di_neg[window - 1..])
        .map(|(pos, neg)| f64::abs(pos - neg) / (pos + neg) * 100.0)
        .collect::<Vec<f64>>();
    iter::once((f64::NAN, f64::NAN, f64::NAN)).chain(izip!(
        di_pos,
        di_neg,
        iter::repeat(f64::NAN)
            .take(window - 1)
            .chain(smooth::wilder(&dx, smoothing).collect::<Vec<f64>>()),
    ))
}

/// Vortex Indicator
///
/// Calculates two lines: VI+ and VI-. The greater the distance between the low of a price bar and
/// the subsequent bar's high, the greater the positive Vortex movement (VM+).
///
/// ## Usage
///
/// When VI+ crosses above VI-, it suggests a uptrend.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Vortex_indicator)
/// [[2]](https://www.investopedia.com/terms/v/vortex-indicator-vi.asp)
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
pub fn vortex<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    iter::repeat((f64::NAN, f64::NAN)).take(window).chain(
        izip!(
            &high[..high.len() - 1],
            &high[1..],
            &low[..low.len() - 1],
            &low[1..],
            &close[..close.len() - 1],
        )
        .map(|(prevh, h, prevl, l, prevc)| {
            let (prevh, h, prevl, l, prevc) = (
                prevh.to_f64().unwrap(),
                h.to_f64().unwrap(),
                prevl.to_f64().unwrap(),
                l.to_f64().unwrap(),
                prevc.to_f64().unwrap(),
            );
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
        .collect::<Vec<(f64, f64)>>(),
    )
}

/// Accumulative Swing Index
///
/// Cumulative sum of Swing Index
///
/// ## Usage
///
/// An increasing value suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/a/asi.asp)
/// [[2]](https://quantstrategy.io/blog/accumulative-swing-index-how-to-trade/)
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
pub fn asi<'a, T: ToPrimitive>(
    open: &'a [T],
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    limit: f64,
) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(_swing(open, high, low, close, limit).scan(0.0, |acc, x| {
        *acc += x;
        Some(*acc)
    }))
}

/// Ulcer Index
///
/// Measures downside risk in terms of both the depth and duration of price declines.
/// The index increases in value as the price moves farther away from a recent high
/// and falls as the price rises to new highs.
///
/// ## Usage
///
/// Value decreases during uptrends.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Ulcer_index)
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
pub fn ulcer<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    let highest = data.windows(window).map(|w| {
        w.iter()
            .fold(f64::NAN, |state, x| state.max(x.to_f64().unwrap()))
    });
    iter::repeat(f64::NAN).take(window - 1).chain(
        smooth::sma(
            &highest
                .zip(data.iter().skip(window - 1))
                .map(|(high, c)| (100.0 * (c.to_f64().unwrap() - high) / high).powi(2))
                .collect::<Vec<f64>>(),
            window,
        )
        .map(|x| x.sqrt())
        .collect::<Vec<f64>>(),
    )
}

/// Supertrend
///
/// Acts as a dynamic level of support or resistance.
///
/// ## Usage
///
/// A value above close line suggests downtrend.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/support/solutions/43000634738-supertrend/)
/// [[2]](https://www.investopedia.com/supertrend-indicator-7976167)
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
pub fn supertrend<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
    multiplier: f64,
) -> impl Iterator<Item = f64> + 'a {
    // TODO: needs a test for when it actually flips to use upper band line
    let tr = _true_range(high, low, close).collect::<Vec<f64>>();
    let atr = iter::once(f64::NAN).chain(smooth::wilder(&tr, window));
    izip!(high, low, close, atr)
        .scan(
            (f64::NAN, f64::NAN, f64::MIN_POSITIVE, 1),
            |state, (h, l, c, tr)| {
                let (h, l, c) = (
                    h.to_f64().unwrap(),
                    l.to_f64().unwrap(),
                    c.to_f64().unwrap(),
                );
                let (prevlower, prevupper, prevc, prevdir) = state;
                let mut lower = (h + l) / 2.0 - multiplier * tr;
                let mut upper = (h + l) / 2.0 + multiplier * tr;
                if prevc > prevlower && *prevlower > lower {
                    lower = *prevlower;
                }
                if prevc < prevupper && *prevupper < upper {
                    upper = *prevupper;
                }
                let dir = if c > *prevupper {
                    1
                } else if c < *prevlower {
                    -1
                } else {
                    *prevdir
                };
                *state = (lower, upper, c, dir);
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
/// ## Usage
///
/// When RWI High is above RWI Low, it suggests an uptrend and strength increases as it goes
/// above 1.
///
/// NOTE: Window includes current price where other libraries use window strictly as lookback.
/// You may need to add 1 to window for comparable behaviour.
///
/// ## Sources
///
/// [[1]](https://www.technicalindicators.net/indicators-technical-analysis/168-rwi-random-walk-index)
/// [[2]](https://www.investopedia.com/terms/r/random-walk-index.asp)
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
pub fn rwi<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    // looks back n number of periods *including* current. other libs may not include current.
    iter::repeat((f64::NAN, f64::NAN)).take(window).chain(
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
                rwi_high = rwi_high.max(
                    (h[window - 1].to_f64().unwrap() - l[window - i].to_f64().unwrap()) / denom,
                );
                rwi_low = rwi_low.max(
                    (h[window - i].to_f64().unwrap() - l[window - 1].to_f64().unwrap()) / denom,
                );
            }
            (rwi_high, rwi_low)
        })
        .collect::<Vec<(f64, f64)>>(),
    )
}

/// Parabolic Stop and Reverse (SAR)
///
/// Calculating the stop for each upcoming period. When the stop is hit you close the
/// current trade and initiate a new trade in the opposite direction.
///
/// ## Usage
///
/// A value above close line suggests downtrend.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/p/parabolicindicator.asp)
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
pub fn psar<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    af: Option<f64>,
    af_max: Option<f64>,
) -> impl Iterator<Item = f64> + 'a {
    let af_inc = af.unwrap_or(0.02);
    let mut af = af_inc;
    let af_max = af_max.unwrap_or(0.2);
    let mut fall = true;
    let mut sar = high[0].to_f64().unwrap();
    let mut ep = low[0].to_f64().unwrap();

    let mut flip: bool;
    let mut result = Vec::with_capacity(high.len() - 1);
    for i in 0..high.len() - 1 {
        sar = sar + af * (ep - sar);
        if fall {
            flip = high[i + 1].to_f64().unwrap() > sar;
            if low[i + 1].to_f64().unwrap() < ep {
                ep = low[i + 1].to_f64().unwrap();
                af = f64::min(af + af_inc, af_max);
            }
            sar = f64::max(
                f64::max(
                    high[std::cmp::max(i, 1) - 1].to_f64().unwrap(),
                    high[i + 1].to_f64().unwrap(),
                ),
                sar,
            )
        } else {
            flip = low[i + 1].to_f64().unwrap() < sar;
            if high[i + 1].to_f64().unwrap() > ep {
                ep = high[i + 1].to_f64().unwrap();
                af = f64::min(af + af_inc, af_max);
            }
            sar = f64::min(
                f64::min(
                    low[std::cmp::max(i, 1) - 1].to_f64().unwrap(),
                    low[i + 1].to_f64().unwrap(),
                ),
                sar,
            )
        }

        if flip {
            sar = ep;
            af = af_inc;
            fall = !fall;
            ep = if fall {
                low[i + 1].to_f64().unwrap()
            } else {
                high[i + 1].to_f64().unwrap()
            };
        }
        result.push(sar);
    }
    iter::once(f64::NAN).chain(result)
}

/// Detrended Price Oscillator
///
/// Measure the difference between a security's price and its trend and attempts to identify
/// cyclicality. It behaves closer to a momentum indicator
///
/// ## Usage
///
/// Suggests uptrend when above zero or when value reaches prior low of oscillator.
///
/// # Examples
///
/// ```
/// use traquer::{trend,smooth};
///
/// trend::dpo(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6, Some(smooth::MaMode::SMA)).collect::<Vec<f64>>();
///
/// ```
pub fn dpo<T: ToPrimitive>(
    data: &[T],
    window: usize,
    mamode: Option<smooth::MaMode>,
) -> impl Iterator<Item = f64> + '_ {
    let ma = smooth::ma(data, window, mamode.unwrap_or(smooth::MaMode::SMA));
    let lag = window / 2 + 1;
    iter::repeat(f64::NAN).take(window - 1).chain(
        data[window - lag - 1..]
            .iter()
            .zip(ma.skip(window - 1))
            .map(|(x, y)| x.to_f64().unwrap() - y),
    )
}

/// Aroon
///
/// Measures the trend strength and direction of an asset based on the time between highs and
/// lows. Consists of two lines that show the time since a recent high or low.
///
/// ## Usage
///
/// When up line is above the down, it suggests an uptrend.
///
/// ## Sources
///
/// [[1]](https://www.tradingview.com/support/solutions/43000501801-aroon/)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::aroon(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn aroon<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    iter::repeat((f64::NAN, f64::NAN)).take(window).chain(
        high.windows(window + 1)
            .zip(low.windows(window + 1))
            .map(move |(h, l)| {
                let (hh, _) =
                    h.iter()
                        .enumerate()
                        .fold((0, h[0].to_f64().unwrap()), |state, (idx, x)| {
                            if x.to_f64().unwrap() >= state.1 {
                                (idx, x.to_f64().unwrap())
                            } else {
                                state
                            }
                        });
                let (ll, _) =
                    l.iter()
                        .enumerate()
                        .fold((0, l[0].to_f64().unwrap()), |state, (idx, x)| {
                            if x.to_f64().unwrap() <= state.1 {
                                (idx, x.to_f64().unwrap())
                            } else {
                                state
                            }
                        });
                (
                    hh as f64 / window as f64 * 100.0,
                    ll as f64 / window as f64 * 100.0,
                )
            }),
    )
}

/// Chandelier Exit
///
/// A stop-loss strategy (trailing stop) that is based on the volatility measured by
/// the Average True Range (ATR) indicator. The Chandelier Exit consists of two lines:
/// one line (Chandelier Exit Long) is used to close long position and the other
/// (Chandelier Exit Short) is used to close short position.
///
/// ## Usage
///
/// Close long position when price drops below Exit Long line.
///
/// ## Sources
///
/// [[1]](https://www.marketvolume.com/technicalanalysis/chandelierexit.asp)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::chandelier(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, Some(3.0)).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn chandelier<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
    multiplier: Option<f64>,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let multiplier = multiplier.unwrap_or(3.0);
    iter::repeat((f64::NAN, f64::NAN)).take(window).chain(
        izip!(
            high.windows(window).skip(1),
            low.windows(window).skip(1),
            atr(high, low, close, window).skip(window)
        )
        .map(move |(h, l, atr)| {
            let hh = h
                .iter()
                .fold(f64::NAN, |state, x| state.max(x.to_f64().unwrap()));
            let ll = l
                .iter()
                .fold(f64::NAN, |state, x| state.min(x.to_f64().unwrap()));
            (hh - multiplier * atr, ll + multiplier * atr)
        }),
    )
}

/// ZigZag Indicator
///
/// A series of trend lines that connect significant peaks and valleys.
///
/// ## Usage
///
/// Used in conjuction with Elliot Wave Theory. Can be used to identify support/resistence
/// but need keep in mind it (may) redraw last trendline as new data comes in.
///
/// ## Sources
///
/// [[1]](https://www.marketvolume.com/technicalanalysis/zig_zag.asp)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::zigzag(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     Some(3.0)).collect::<Vec<f64>>();
///
/// ```
pub fn zigzag<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    distance: Option<f64>,
) -> impl Iterator<Item = f64> + 'a {
    let distance = distance.unwrap_or(5.0);
    let mut result: Vec<f64> = vec![f64::NAN; high.len()];

    let mut startxy: (usize, f64) = (0, low[0].to_f64().unwrap());
    let mut endxy: (usize, f64) = (high.len() - 1, high.last().unwrap().to_f64().unwrap());

    // get initial trend
    result[0] = low[0].to_f64().unwrap();
    for idx in 1..high.len() {
        if (high[0].to_f64().unwrap() - low[idx].to_f64().unwrap()) / high[0].to_f64().unwrap()
            * 100.0
            >= distance
        {
            result[0] = high[0].to_f64().unwrap();
            startxy = (0, high[0].to_f64().unwrap());
            endxy = (idx, low[idx].to_f64().unwrap());
            break;
        } else if (high[idx].to_f64().unwrap() - low[0].to_f64().unwrap())
            / low[0].to_f64().unwrap()
            * 100.0
            >= distance
        {
            result[0] = low[0].to_f64().unwrap();
            startxy = (0, low[0].to_f64().unwrap());
            endxy = (idx, high[idx].to_f64().unwrap());
            break;
        }
    }

    // handle remaining data
    if endxy.0 < high.len() - 1 {
        for i in endxy.0..high.len() {
            if endxy.1 > startxy.1 && high[i].to_f64().unwrap() > endxy.1 {
                // same trend but greater distance, extend trend.
                endxy = (i, high[i].to_f64().unwrap());
            } else if endxy.1 < startxy.1
                && (high[i].to_f64().unwrap() - endxy.1) / high[i].to_f64().unwrap() * 100.0
                    >= distance
            {
                // opposite trend and significant distance, capture pivot and start new trend.
                result[endxy.0] = endxy.1;
                startxy = endxy;
                endxy = (i, high[i].to_f64().unwrap());
            } else if endxy.1 < startxy.1 && low[i].to_f64().unwrap() < endxy.1 {
                endxy = (i, low[i].to_f64().unwrap());
            } else if endxy.1 > startxy.1
                && (endxy.1 - low[i].to_f64().unwrap()) / low[i].to_f64().unwrap() * 100.0
                    >= distance
            {
                result[endxy.0] = endxy.1;
                startxy = endxy;
                endxy = (i, low[i].to_f64().unwrap());
            }
        }
    }

    // handle last trend
    if result.last().unwrap().is_nan() {
        let result_size = result.len();
        if startxy.1 > endxy.1 {
            result[result_size - 1] = high.last().unwrap().to_f64().unwrap();
        } else {
            result[result_size - 1] = low.last().unwrap().to_f64().unwrap();
        }
    }
    result.into_iter()
}

/// Decay
///
/// Propagates signals from the past into the future. Minimises lower prices.
///
/// ## Usage
///
/// ¯\\_(ツ)_/¯
///
/// ## Sources
///
/// [[1]](https://tulipindicators.org/decay)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::decay(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3).collect::<Vec<f64>>();
///
/// ```
pub fn decay<T: ToPrimitive>(data: &[T], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.iter().scan(0.0, move |state, x| {
        *state = (*state - 1.0 / window as f64)
            .max(x.to_f64().unwrap())
            .max(0.0);
        Some(*state)
    })
}

/// Chande Kroll Stop
///
/// Calculated using a combination of the highest high and lowest low prices over a specific
/// lookback period and the instrument’s ATR. Produces two stop-loss levels.
///
/// ## Usage
///
/// Buy signal when price crosses above the long and short lines.
///
/// ## Sources
///
/// [[1]](https://blog.xcaldata.com/chande-kroll-stop-for-mastering-risk-management-in-stock-market/)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::cks(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     5, 3, Some(1.0)).collect::<Vec<(f64, f64)>>();
///
/// ```
pub fn cks<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    win1: usize,
    win2: usize,
    multiplier: Option<f64>,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let multiplier = multiplier.unwrap_or(1.0);
    iter::repeat((f64::NAN, f64::NAN))
        .take(win1 + (win2 - 1))
        .chain(
            izip!(
                high[1..].windows(win1),
                low[1..].windows(win1),
                atr(high, low, close, win1).skip(win1)
            )
            .map(|(h, l, tr)| {
                (
                    h.iter()
                        .fold(f64::NAN, |state, x| state.max(x.to_f64().unwrap()))
                        - multiplier * tr,
                    l.iter()
                        .fold(f64::NAN, |state, x| state.min(x.to_f64().unwrap()))
                        + multiplier * tr,
                )
            })
            .collect::<Vec<(f64, f64)>>()
            .windows(win2)
            .map(|w| {
                let mut hh: f64 = 0.0;
                let mut ll: f64 = f64::MAX;
                for (h, l) in w {
                    hh = hh.max(*h);
                    ll = ll.min(*l);
                }
                (hh, ll)
            })
            .collect::<Vec<(f64, f64)>>(),
        )
}

/// ATR Trailing Stop
///
/// Adjusts the stop-loss orders according to the asset’s volatility.
///
/// ## Usage
///
/// Buy signal when prices cross above ATR trailing stop line.
///
/// ## Sources
///
/// [[1]](https://www.incrediblecharts.com/indicators/atr_average_true_range_trailing_stops.php)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::atr_stop(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, Some(1.0)).collect::<Vec<f64>>();
///
/// ```
pub fn atr_stop<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    window: usize,
    multiplier: Option<f64>,
) -> impl Iterator<Item = f64> + 'a {
    let multiplier = multiplier.unwrap_or(3.0);

    iter::repeat(f64::NAN).take(window).chain(
        izip!(
            close.windows(window).skip(1),
            atr(high, low, close, window).skip(window)
        )
        .scan((true, 0.0), move |state, (c, atr)| {
            // state = (uptrend, prev_band)
            let upper = c[window - 1].to_f64().unwrap() + multiplier * atr;
            let lower = c[window - 1].to_f64().unwrap() - multiplier * atr;
            let mut band: f64 = state.1;
            if state.0 {
                band = band.max(lower);
                *state = if c[window - 1].to_f64().unwrap() < band {
                    (false, upper)
                } else {
                    (state.0, band)
                };
            } else {
                band = band.min(upper);
                *state = if c[window - 1].to_f64().unwrap() > band {
                    (true, lower)
                } else {
                    (state.0, band)
                };
            }
            Some(state.1)
        }),
    )
}

/// Williams Alligator
///
/// Three smoothed moving averages of median price known as the Jaw, Teeth, and Lips.
/// Follows the premise that financial markets and individual securities trend
/// just 15% to 30% of the time while grinding through sideways ranges the
/// other 70% to 85% of the time.
///
/// ## Usage
///
/// Lips line crossing above the other lines signals a buying opportunity.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/articles/trading/072115/exploring-williams-alligator-indicator.asp)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::alligator(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0,2.0,3.0,4.0,5.0,6.0],
///     6, 4, 4, 3, 3, 2).collect::<Vec<(f64, f64, f64)>>();
///
/// ```
pub fn alligator<T: ToPrimitive>(
    data: &[T],
    jaw_win: usize,
    jaw_offset: usize,
    teeth_win: usize,
    teeth_offset: usize,
    lips_win: usize,
    lips_offset: usize,
) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
    let max_offset = jaw_offset.max(teeth_offset).max(lips_offset);
    let jaw = iter::repeat(f64::NAN)
        .take(jaw_offset)
        .chain(smooth::wilder(data, jaw_win))
        .chain(iter::repeat(f64::NAN).take(max_offset - jaw_offset));
    let teeth = iter::repeat(f64::NAN)
        .take(teeth_offset)
        .chain(smooth::wilder(data, teeth_win))
        .chain(iter::repeat(f64::NAN).take(max_offset - teeth_offset));
    let lips = iter::repeat(f64::NAN)
        .take(lips_offset)
        .chain(smooth::wilder(data, lips_win))
        .chain(iter::repeat(f64::NAN).take(max_offset - lips_offset));
    izip!(jaw, teeth, lips)
}

/// Ichimoku Cloud
///
/// Composed of five lines or calculations, two of which comprise a cloud.
///
/// ## Usage
///
/// Suggests an uptrend when price is above the cloud.
///
/// ## Sources
///
/// [[1]](https://www.investopedia.com/terms/i/ichimoku-cloud.asp)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::ichimoku(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3,2,2,2).collect::<Vec<(f64,f64,f64,f64,f64)>>();
///
/// ```
pub fn ichimoku<'a, T: ToPrimitive>(
    high: &'a [T],
    low: &'a [T],
    close: &'a [T],
    conversion_win: usize,
    base_win: usize,
    lead_win: usize,
    lag_win: usize,
) -> impl Iterator<Item = (f64, f64, f64, f64, f64)> + 'a {
    let c_line = iter::repeat(f64::NAN)
        .take(conversion_win - 1)
        .chain(
            high.windows(conversion_win)
                .zip(low.windows(conversion_win))
                .map(|(h, l)| {
                    let hh = h
                        .iter()
                        .fold(f64::NAN, |state, x| state.max(x.to_f64().unwrap()));
                    let ll = l
                        .iter()
                        .fold(f64::NAN, |state, x| state.min(x.to_f64().unwrap()));
                    (hh + ll) / 2.0
                }),
        )
        .collect::<Vec<f64>>();
    let b_line = iter::repeat(f64::NAN)
        .take(base_win - 1)
        .chain(
            high.windows(base_win)
                .zip(low.windows(base_win))
                .map(|(h, l)| {
                    let hh = h
                        .iter()
                        .fold(f64::NAN, |state, x| state.max(x.to_f64().unwrap()));
                    let ll = l
                        .iter()
                        .fold(f64::NAN, |state, x| state.min(x.to_f64().unwrap()));
                    (hh + ll) / 2.0
                }),
        )
        .collect::<Vec<f64>>();
    let lead_b = iter::repeat(f64::NAN).take(lead_win - 1 + base_win).chain(
        high.windows(lead_win)
            .zip(low.windows(lead_win))
            .map(|(h, l)| {
                let hh = h
                    .iter()
                    .fold(f64::NAN, |state, x| state.max(x.to_f64().unwrap()));
                let ll = l
                    .iter()
                    .fold(f64::NAN, |state, x| state.min(x.to_f64().unwrap()));
                (hh + ll) / 2.0
            }),
    );
    let lead_a = iter::repeat(f64::NAN)
        .take(base_win)
        .chain(c_line.iter().zip(&b_line).map(|(c, b)| (c + b) / 2.0))
        .collect::<Vec<f64>>();
    let lag = close
        .iter()
        .skip(lag_win)
        .filter_map(|x| x.to_f64())
        .chain(iter::repeat(f64::NAN).take(lag_win + base_win));
    izip!(
        c_line
            .into_iter()
            .chain(iter::repeat(f64::NAN).take(base_win)),
        b_line
            .into_iter()
            .chain(iter::repeat(f64::NAN).take(base_win)),
        lead_a,
        lead_b,
        lag
    )
}

fn hurst_rs<T: ToPrimitive>(data: &[T]) -> f64 {
    let avg = data.iter().filter_map(|x| x.to_f64()).sum::<f64>() / data.len() as f64;
    let mut max_z = f64::MIN;
    let mut min_z = f64::MAX;
    let mut dev_cumsum = 0.0;
    let mut var = 0.0;
    for x in data {
        let dev = x.to_f64().unwrap() - avg;
        var += dev.powi(2);
        dev_cumsum += dev;
        max_z = max_z.max(dev_cumsum);
        min_z = min_z.min(dev_cumsum);
    }

    let r = max_z - min_z;
    let s = (var / (data.len() - 1) as f64).sqrt();
    r / s
}

fn linreg(x: &[f64], y: &[f64]) -> f64 {
    let x_avg = x.iter().fold(0.0, |acc, &x| acc + x) / x.len() as f64;
    let y_avg = y.iter().fold(0.0, |acc, &y| acc + y) / y.len() as f64;
    let mut covar = 0.0;
    let mut var = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        covar += (xi - x_avg) * (yi - y_avg);
        var += (xi - x_avg).powi(2);
    }
    covar / var
}

/// Hurst Exponent
///
/// Measures the long-term memory of time series, and the rate at which these decrease as the
/// lag between pairs of values increases.
///
/// Configured to accept delta / return values as input rather than price. Window size should be
/// significantly larger than min_win size to ensure enough datapoints are computed for estimation
/// or else values less than 0 or greater than 1 may result.
///
/// ## Usage
///
/// A value above 0.5 suggests a positive autocorrelation or that there is a trend and
/// it will continue (either up or down). A value below 0.5 suggests it will mean-revert
/// or that it will break trend.
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Hurst_exponent)
/// [[2]](https://github.com/Mottl/hurst/)
/// [[3]](https://mahowald.github.io/hurst/)
///
/// # Examples
///
/// ```
/// use traquer::trend;
///
/// trend::hurst(&vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0], 5, None).collect::<Vec<f64>>();
///
/// ```
pub fn hurst<T: ToPrimitive>(
    data: &[T],
    window: usize,
    min_win: Option<usize>,
) -> impl Iterator<Item = f64> + '_ {
    // create sub window sizes
    let mut win_sizes = Vec::new();
    let mut win = (min_win.unwrap_or(4) as f64).log10();
    loop {
        let win_size = 10.0_f64.powf(win) as usize;
        if win_size >= window {
            break;
        }
        win_sizes.push(win_size);
        win += 0.25;
    }
    win_sizes.push(window);
    let win_size_log10 = win_sizes
        .iter()
        .map(|&x| (x as f64).log10())
        .collect::<Vec<_>>();

    iter::repeat(f64::NAN)
        .take(window - 1)
        .chain(data.windows(window).map(move |w| {
            let mut rs_vals = Vec::new();
            for &x in win_sizes.iter() {
                let chunks = w.chunks_exact(x);
                let cnt = chunks.len();
                rs_vals.push((chunks.fold(0.0, |acc, x| acc + hurst_rs(x)) / cnt as f64).log10());
            }
            linreg(&win_size_log10, &rs_vals)
        }))
}
