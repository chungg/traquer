use std::iter;

use itertools::izip;

use crate::smooth;
use crate::trend::_true_range;

/// relative strength index
/// https://www.investopedia.com/terms/r/rsi.asp
pub fn rsi(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let (gain, loss): (Vec<f64>, Vec<f64>) = data[1..]
        .iter()
        .zip(data[..data.len() - 1].iter())
        .map(|(curr, prev)| (f64::max(0.0, curr - prev), f64::min(0.0, curr - prev).abs()))
        .unzip();
    smooth::wilder(&gain, window)
        .zip(smooth::wilder(&loss, window))
        .map(|(g, l)| 100.0 * g / (g + l))
        .collect::<Vec<f64>>()
        .into_iter()
}

/// moving average convergence/divergence
/// https://www.investopedia.com/terms/m/macd.asp
pub fn macd(close: &[f64], short: usize, long: usize) -> impl Iterator<Item = f64> + '_ {
    let short_ma = smooth::ewma(close, short);
    let long_ma = smooth::ewma(close, long);
    short_ma.skip(long - short).zip(long_ma).map(|(x, y)| x - y)
}

/// chande momentum oscillator
/// https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
pub fn cmo(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    smooth::_cmo(data, window).map(|x| x * 100.0)
}

/// elder ray
/// https://www.investopedia.com/articles/trading/03/022603.asp
/// returns tuple of bull power vec and bear power vec
pub fn elder_ray<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let close_ma = smooth::ewma(close, window);
    izip!(
        high.iter().skip(window - 1),
        low.iter().skip(window - 1),
        close_ma
    )
    .map(|(h, l, c)| (h - c, l - c))
}

/// williams alligator
/// https://www.investopedia.com/articles/trading/072115/exploring-williams-alligator-indicator.asp
pub fn alligator(_data: &[f64]) {}

/// chaikin volatility
/// https://www.tradingview.com/chart/AUDUSD/gjfxqWqW-What-Is-a-Chaikin-Volatility-Indicator-in-Trading/
/// https://theforexgeek.com/chaikins-volatility-indicator/
pub fn cvi<'a>(
    high: &'a [f64],
    low: &'a [f64],
    window: usize,
    rate_of_change: usize,
) -> impl Iterator<Item = f64> + 'a {
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
    .into_iter()
}

/// Williams Percent Range
/// https://www.investopedia.com/terms/w/williamsr.asp
pub fn wpr<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
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
}

/// percent price oscillator
/// pass in any data (close, high, low, etc...), and two window ranges
pub fn ppo(data: &[f64], short: usize, long: usize) -> impl Iterator<Item = f64> + '_ {
    let short_ma = smooth::ewma(data, short);
    let long_ma = smooth::ewma(data, long);
    short_ma
        .skip(long - short)
        .zip(long_ma)
        .map(|(x, y)| 100.0 * (x / y - 1.0))
}

/// Absolute Price Oscillator
/// https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo
pub fn apo(data: &[f64], short: usize, long: usize) -> impl Iterator<Item = f64> + '_ {
    let short_ma = smooth::ewma(data, short);
    let long_ma = smooth::ewma(data, long);
    short_ma.skip(long - short).zip(long_ma).map(|(x, y)| x - y)
}

/// Detrended Price Oscillator
pub fn dpo(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let ma = smooth::sma(data, window);
    let lag = window / 2 + 1;
    data[window - lag - 1..].iter().zip(ma).map(|(x, y)| x - y)
}

/// ultimate oscillator
/// https://www.investopedia.com/terms/u/ultimateoscillator.asp
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
        .into_iter()
}

/// pretty good oscillator
/// https://library.tradingtechnologies.com/trade/chrt-ti-pretty-good-oscillator.html
pub fn pgo<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let tr = _true_range(high, low, close).collect::<Vec<f64>>();
    let atr = smooth::ewma(&tr, window);
    let sma_close = smooth::sma(close, window);
    izip!(close.iter().skip(window), sma_close.skip(1), atr)
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
/// https://www.investopedia.com/terms/a/asi.asp
/// https://quantstrategy.io/blog/accumulative-swing-index-how-to-trade/
pub fn si<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    limit: f64,
) -> impl Iterator<Item = f64> + 'a {
    _swing(open, high, low, close, limit)
}

/// Triple Exponential Average
/// https://www.investopedia.com/terms/t/trix.asp
pub fn trix(close: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
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
        .into_iter()
}

/// trend intensity index
/// https://www.marketvolume.com/technicalanalysis/trendintensityindex.asp
pub fn tii(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    smooth::sma(data, window)
        .zip(&data[(window - 1)..])
        .map(|(avg, actual)| {
            let dev: f64 = actual - avg;
            (dev.max(0.0), dev.min(0.0).abs())
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
        .into_iter()
}

/// Stochastic Oscillator
/// https://www.investopedia.com/articles/technical/073001.asp
pub fn stochastic<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = (f64, f64)> + 'a {
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
    izip!(fast_k, iter::repeat(f64::NAN).take(3 - 1).chain(k))
}

fn _stc(series: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
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
    .into_iter()
}

/// Schaff Trend Cycle
/// https://www.investopedia.com/articles/forex/10/schaff-trend-cycle-indicator.asp
/// https://www.stockmaniacs.net/schaff-trend-cycle-indicator/
pub fn stc(
    close: &[f64],
    window: usize,
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
    let series = macd(close, short, long);
    _stc(
        &_stc(&series.collect::<Vec<f64>>(), window).collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Inertia
/// https://theforexgeek.com/inertia-indicator/
pub fn inertia(close: &[f64], window: usize, smoothing: usize) -> impl Iterator<Item = f64> + '_ {
    smooth::lrf(
        &relative_vol(close, window, window).collect::<Vec<f64>>(),
        smoothing,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Relative Volatility
/// https://www.tradingview.com/support/solutions/43000594684-relative-volatility-index/
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
    smooth::wilder(&gain, smoothing)
        .zip(smooth::wilder(&loss, smoothing))
        .map(|(g, l)| 100.0 * g / (g + l))
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Relative Vigor
/// https://www.investopedia.com/terms/r/relative_vigor_index.asp
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
    smooth::sma(&numerator, window)
        .zip(smooth::sma(&denominator, window))
        .map(|(n, d)| n / d)
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Elhers Fisher Transform
/// https://www.investopedia.com/terms/f/fisher-transform.asp
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
                .min(0.999999)
                .max(-0.999999);
            let result = 0.5 * ((1.0 + transform) / (1.0 - transform)).ln() + 0.5 * state.1;
            *state = (transform, result);
            Some(state.1)
        })
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Rainbow Oscillator
/// https://www.tradingview.com/script/gWYg0ti0-Indicators-Rainbow-Charts-Oscillator-Binary-Wave-and-MAs/
pub fn rainbow(
    data: &[f64],
    window: usize,
    lookback: usize,
) -> impl Iterator<Item = (f64, f64)> + '_ {
    let mut smas = Vec::with_capacity(10);
    smas.push(smooth::sma(data, window).collect::<Vec<f64>>());
    for _ in 1..10 {
        smas.push(
            iter::repeat(f64::NAN)
                .take(window - 1)
                .chain(smooth::sma(&smas[smas.len() - 1], window))
                .collect::<Vec<f64>>(),
        );
    }
    ((window - 1) * 10..data.len()).map(move |i| {
        let mut total: f64 = 0.0;
        let mut hsma = f64::MIN;
        let mut lsma = f64::MAX;
        for sma in smas.iter() {
            let val = sma[i - (window - 1)];
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
    })
}

/// Coppock Curve
/// https://www.investopedia.com/terms/c/coppockcurve.asp
pub fn coppock(
    data: &[f64],
    window: usize,
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + '_ {
    smooth::wma(
        &(long..data.len())
            .map(|x| 100.0 * (data[x] / data[x - short] + data[x] / data[x - long] - 2.0))
            .collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Rate of Change
/// https://www.investopedia.com/terms/p/pricerateofchange.asp.
pub fn roc(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.windows(window)
        .map(|w| 100.0 * (w[w.len() - 1] / w[0] - 1.0))
}

/// Choppiness Index
/// https://www.tradingview.com/support/solutions/43000501980-choppiness-index-chop/
pub fn chop<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
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
    .collect::<Vec<f64>>()
    .into_iter()
}

/// Balance of Power
/// https://www.tradingview.com/support/solutions/43000589100-balance-of-power-bop/
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
/// https://www.investopedia.com/terms/d/disparityindex.asp
pub fn disparity(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.iter()
        .skip(window - 1)
        .zip(smooth::ewma(data, window))
        .map(|(x, ma)| 100.0 * (x - ma) / ma)
}
