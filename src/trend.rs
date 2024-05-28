use std::iter;

use itertools::{izip, multiunzip};

use crate::momentum::_swing;
use crate::smooth;

/// quick stick
/// https://www.investopedia.com/terms/q/qstick.asp
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

/// shinohara intensity ratio
/// https://www.sevendata.co.jp/shihyou/technical/shinohara.html
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

/// average directional index
/// https://www.investopedia.com/terms/a/adx.asp
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

/// centre of gravity
/// https://www.stockmaniacs.net/center-of-gravity-indicator/
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

/// vortex
/// https://www.investopedia.com/terms/v/vortex-indicator-vi.asp
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

/// vertical horizontal filter
/// https://www.upcomingtrader.com/blog/the-vertical-horizontal-filter-a-traders-guide-to-market-phases/
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
/// https://www.investopedia.com/terms/a/asi.asp
/// https://quantstrategy.io/blog/accumulative-swing-index-how-to-trade/
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
/// https://en.wikipedia.org/wiki/Ulcer_index
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

/// supertrend
/// https://www.tradingview.com/support/solutions/43000634738-supertrend/
/// https://www.investopedia.com/supertrend-indicator-7976167
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
/// https://www.technicalindicators.net/indicators-technical-analysis/168-rwi-random-walk-index
/// https://www.investopedia.com/terms/r/random-walk-index.asp
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
/// https://tradingliteracy.com/psychological-line-indicator/
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
/// https://www.investopedia.com/terms/m/mass-index.asp
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
/// https://www.investopedia.com/terms/k/keltnerchannel.asp
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
/// https://library.tradingtechnologies.com/trade/chrt-ti-gopalakrishnan-range-index.html
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

/// true range
/// https://www.investopedia.com/terms/a/atr.asp
pub fn tr<'a>(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    _true_range(high, low, close)
}

/// average true range
/// https://www.investopedia.com/terms/a/atr.asp
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

/// typical price
/// https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/typical-price
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
