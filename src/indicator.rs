use itertools::{izip, multiunzip};

use crate::smooth;

/// quick stick
/// https://www.investopedia.com/terms/q/qstick.asp
pub fn qstick(open: &[f64], close: &[f64], window: usize) -> Vec<f64> {
    let q = close
        .iter()
        .zip(open.iter())
        .map(|(c, o)| c - o)
        .collect::<Vec<f64>>();
    smooth::ewma(&q, window).collect::<Vec<f64>>()
}

/// shinohara intensity ratio
/// https://www.sevendata.co.jp/shihyou/technical/shinohara.html
pub fn shinohara(high: &[f64], low: &[f64], close: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let high_win = high
        .windows(period)
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    let low_win = low
        .windows(period)
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    let close_win = close
        .windows(period)
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
    (strong_ratio, weak_ratio)
}

/// average directional index
/// https://www.investopedia.com/terms/a/adx.asp
pub fn adx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    smoothing: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
    let atr = smooth::wilder(&tr, period).collect::<Vec<f64>>();
    let di_pos = izip!(smooth::wilder(&dm_pos, period), &atr)
        .map(|(di, tr)| di / tr * 100.0)
        .collect::<Vec<f64>>();
    let di_neg = izip!(smooth::wilder(&dm_neg, period), &atr)
        .map(|(di, tr)| di / tr * 100.0)
        .collect::<Vec<f64>>();
    let dx = izip!(&di_pos, &di_neg)
        .map(|(pos, neg)| f64::abs(pos - neg) / (pos + neg) * 100.0)
        .collect::<Vec<f64>>();
    (
        di_pos,
        di_neg,
        smooth::wilder(&dx, smoothing).collect::<Vec<f64>>(),
    )
}

/// relative strength index
/// https://www.investopedia.com/terms/r/rsi.asp
pub fn rsi(data: &[f64], window: usize) -> Vec<f64> {
    let (gain, loss): (Vec<f64>, Vec<f64>) = data[1..]
        .iter()
        .zip(data[..data.len() - 1].iter())
        .map(|(curr, prev)| (f64::max(0.0, curr - prev), f64::min(0.0, curr - prev).abs()))
        .unzip();
    smooth::wilder(&gain, window)
        .zip(smooth::wilder(&loss, window))
        .map(|(g, l)| 100.0 * g / (g + l))
        .collect::<Vec<f64>>()
}

/// moving average convergence/divergence
/// https://www.investopedia.com/terms/m/macd.asp
pub fn macd(close: &[f64], short: usize, long: usize) -> Vec<f64> {
    let short_ma = smooth::ewma(close, short);
    let long_ma = smooth::ewma(close, long);
    short_ma
        .skip(long - short)
        .zip(long_ma)
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

/// chande momentum oscillator
/// https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
pub fn cmo(data: &[f64], window: usize) -> Vec<f64> {
    smooth::_cmo(data, window)
        .map(|x| x * 100.0)
        .collect::<Vec<f64>>()
}

/// centre of gravity
/// https://www.stockmaniacs.net/center-of-gravity-indicator/
pub fn cog(data: &[f64], window: usize) -> Vec<f64> {
    let weights: Vec<f64> = (1..=window).map(|x| x as f64).collect();
    data.windows(window)
        .map(|w| {
            -w.iter()
                .rev()
                .zip(weights.iter())
                .map(|(e, i)| e * i)
                .sum::<f64>()
                / w.iter().sum::<f64>()
        })
        .collect::<Vec<f64>>()
}

/// elder ray
/// https://www.investopedia.com/articles/trading/03/022603.asp
/// returns tuple of bull power vec and bear power vec
pub fn elder_ray(high: &[f64], low: &[f64], close: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let close_ma = smooth::ewma(close, window);
    izip!(
        high.iter().skip(window - 1),
        low.iter().skip(window - 1),
        close_ma
    )
    .map(|(h, l, c)| (h - c, l - c))
    .unzip()
}

/// williams alligator
/// https://www.investopedia.com/articles/trading/072115/exploring-williams-alligator-indicator.asp
pub fn alligator(_data: &[f64]) {}

/// chaikin volatility
/// https://www.tradingview.com/chart/AUDUSD/gjfxqWqW-What-Is-a-Chaikin-Volatility-Indicator-in-Trading/
/// https://theforexgeek.com/chaikins-volatility-indicator/
pub fn cvi(high: &[f64], low: &[f64], window: usize, rate_of_change: usize) -> Vec<f64> {
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
    .collect::<Vec<f64>>()
}

/// Williams Percent Range
/// https://www.investopedia.com/terms/w/williamsr.asp
pub fn wpr(high: &[f64], low: &[f64], close: &[f64], window: usize) -> Vec<f64> {
    izip!(
        high.windows(window),
        low.windows(window),
        &close[(window - 1)..]
    )
    .map(|(h, l, c)| {
        let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
        let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
        -100.0 * ((hh - c) / (hh - ll))
    })
    .collect::<Vec<f64>>()
}

/// vortex
/// https://www.investopedia.com/terms/v/vortex-indicator-vi.asp
pub fn vortex(high: &[f64], low: &[f64], close: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
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
    .unzip()
}

/// percent price oscillator
/// pass in any data (close, high, low, etc...), and two window ranges
pub fn ppo(data: &[f64], short: usize, long: usize) -> Vec<f64> {
    let short_ma = smooth::ewma(data, short);
    let long_ma = smooth::ewma(data, long);
    short_ma
        .skip(long - short)
        .zip(long_ma)
        .map(|(x, y)| 100.0 * (x / y - 1.0))
        .collect::<Vec<f64>>()
}

/// Absolute Price Oscillator
/// https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo
pub fn apo(data: &[f64], short: usize, long: usize) -> Vec<f64> {
    let short_ma = smooth::ewma(data, short);
    let long_ma = smooth::ewma(data, long);
    short_ma
        .skip(long - short)
        .zip(long_ma)
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

/// Detrended Price Oscillator
pub fn dpo(data: &[f64], window: usize) -> Vec<f64> {
    let ma = smooth::sma(data, window);
    let lag = window / 2 + 1;
    data[window - lag - 1..]
        .iter()
        .zip(ma)
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

/// vertical horizontal filter
/// https://www.upcomingtrader.com/blog/the-vertical-horizontal-filter-a-traders-guide-to-market-phases/
pub fn vhf(high: &[f64], low: &[f64], close: &[f64], window: usize) -> Vec<f64> {
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
}

/// ultimate oscillator
/// https://www.investopedia.com/terms/u/ultimateoscillator.asp
pub fn ultimate(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    win1: usize,
    win2: usize,
    win3: usize,
) -> Vec<f64> {
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
        .collect::<Vec<f64>>()
}

/// pretty good oscillator
/// https://library.tradingtechnologies.com/trade/chrt-ti-pretty-good-oscillator.html
pub fn pgo(high: &[f64], low: &[f64], close: &[f64], window: usize) -> Vec<f64> {
    let tr = _true_range(high, low, close).collect::<Vec<f64>>();
    let atr = smooth::ewma(&tr, window);
    let sma_close = smooth::sma(close, window);
    izip!(close.iter().skip(window), sma_close.skip(1), atr)
        .map(|(c, c_ma, tr_ma)| (c - c_ma) / tr_ma)
        .collect::<Vec<f64>>()
}

fn _swing<'a>(
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
/// https://www.investopedia.com/terms/a/asi.asp
/// https://quantstrategy.io/blog/accumulative-swing-index-how-to-trade/
pub fn si(open: &[f64], high: &[f64], low: &[f64], close: &[f64], limit: f64) -> Vec<f64> {
    _swing(open, high, low, close, limit).collect::<Vec<f64>>()
}

/// Accumulative Swing Index
/// https://www.investopedia.com/terms/a/asi.asp
/// https://quantstrategy.io/blog/accumulative-swing-index-how-to-trade/
pub fn asi(open: &[f64], high: &[f64], low: &[f64], close: &[f64], limit: f64) -> Vec<f64> {
    _swing(open, high, low, close, limit)
        .scan(0.0, |acc, x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<f64>>()
}

/// Ulcer Index
/// https://en.wikipedia.org/wiki/Ulcer_index
pub fn ulcer(data: &[f64], window: usize) -> Vec<f64> {
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
}

fn _true_range<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
) -> impl Iterator<Item = f64> + 'a {
    izip!(&high[1..], &low[1..], &close[..close.len() - 1])
        .map(|(h, l, prevc)| (h - l).max(f64::abs(h - prevc)).max(f64::abs(l - prevc)))
}

/// true range
/// https://www.investopedia.com/terms/a/atr.asp
pub fn tr(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    _true_range(high, low, close).collect::<Vec<f64>>()
}

/// typical price
/// https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/typical-price
pub fn hlc3(high: &[f64], low: &[f64], close: &[f64], window: usize) -> Vec<f64> {
    smooth::sma(
        &izip!(high, low, close)
            .map(|(h, l, c)| (h + l + c) / 3.0)
            .collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>()
}

/// Triple Exponential Average
/// https://www.investopedia.com/terms/t/trix.asp
pub fn trix(close: &[f64], window: usize) -> Vec<f64> {
    let ema3 = smooth::ewma(
        &smooth::ewma(&smooth::ewma(close, window).collect::<Vec<f64>>(), window)
            .collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>();
    ema3[..ema3.len() - 1]
        .iter()
        .zip(&ema3[1..])
        .map(|(prev, curr)| 100.0 * (curr - prev) / prev)
        .collect::<Vec<f64>>()
}

/// trend intensity index
/// https://www.marketvolume.com/technicalanalysis/trendintensityindex.asp
pub fn tii(data: &[f64], window: usize) -> Vec<f64> {
    smooth::sma(data, window)
        .zip(&data[(window - 1)..])
        .map(|(avg, actual)| {
            let dev: f64 = actual - avg;
            let pos_dev = if dev > 0.0 { dev } else { 0.0 };
            let neg_dev = if dev < 0.0 { dev.abs() } else { 0.0 };
            (pos_dev, neg_dev)
        })
        .collect::<Vec<(f64, f64)>>()
        .windows(window.div_ceil(2))
        .map(|w| {
            let mut sd_pos = 0.0;
            let mut sd_neg = 0.0;
            for (pos_dev, neg_dev) in w {
                sd_pos += pos_dev;
                sd_neg += neg_dev;
            }
            100.0 * sd_pos / (sd_pos + sd_neg)
        })
        .collect::<Vec<f64>>()
}

/// supertrend
/// https://www.tradingview.com/support/solutions/43000634738-supertrend/
/// https://www.investopedia.com/supertrend-indicator-7976167
pub fn supertrend(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    multiplier: f64,
) -> Vec<f64> {
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
}

/// Stochastic Oscillator
/// https://www.investopedia.com/articles/technical/073001.asp
pub fn stochastic(high: &[f64], low: &[f64], close: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
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
        3,
    )
    .collect::<Vec<f64>>();
    let k = smooth::sma(&fast_k, 3).collect::<Vec<f64>>();
    (fast_k, k)
}

fn _stc(series: &[f64], window: usize) -> Vec<f64> {
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
        2,
    )
    .collect::<Vec<f64>>()
}

/// Schaff Trend Cycle
/// https://www.investopedia.com/articles/forex/10/schaff-trend-cycle-indicator.asp
/// https://www.stockmaniacs.net/schaff-trend-cycle-indicator/
pub fn stc(close: &[f64], window: usize, short: usize, long: usize) -> Vec<f64> {
    let series = macd(close, short, long);
    _stc(&_stc(&series, window), window)
}

/// Relative Volatility
/// https://www.tradingview.com/support/solutions/43000594684-relative-volatility-index/
pub fn relative_vol(close: &[f64], window: usize, smoothing: usize) -> Vec<f64> {
    let stdev = smooth::std_dev(close, window).collect::<Vec<f64>>();
    let (gain, loss): (Vec<f64>, Vec<f64>) = izip!(
        &stdev,
        &close[close.len() - stdev.len()..],
        &close[close.len() - stdev.len() - 1..close.len() - 1]
    )
    .map(|(std, curr, prev)| {
        (
            f64::max(0.0, f64::max(0.0, curr - prev) * std / (curr - prev).abs()),
            f64::max(
                0.0,
                f64::min(0.0, curr - prev).abs() * std / (curr - prev).abs(),
            ),
        )
    })
    .unzip();
    smooth::wilder(&gain, smoothing)
        .zip(smooth::wilder(&loss, smoothing))
        .map(|(g, l)| 100.0 * g / (g + l))
        .collect::<Vec<f64>>()
}

/// Relative Vigor
/// https://www.investopedia.com/terms/r/relative_vigor_index.asp
pub fn relative_vigor(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
) -> Vec<f64> {
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
    smooth::sma(&numerator, window)
        .collect::<Vec<f64>>()
        .iter()
        .zip(smooth::sma(&denominator, window))
        .map(|(n, d)| n / d)
        .collect::<Vec<f64>>()
}

/// Random Walk Index
/// https://www.technicalindicators.net/indicators-technical-analysis/168-rwi-random-walk-index
/// https://www.investopedia.com/terms/r/random-walk-index.asp
pub fn rwi(high: &[f64], low: &[f64], close: &[f64], window: usize) -> Vec<(f64, f64)> {
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
}
