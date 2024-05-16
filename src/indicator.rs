use itertools::{izip, multiunzip};

use crate::smooth;

fn vforce(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    izip!(&high[1..], &low[1..], &close[1..], &volume[1..])
        .scan((high[0], low[0], close[0], 99), |state, (h, l, c, v)| {
            let trend: i8 = {
                if h + l + c > state.0 + state.1 + state.2 {
                    1
                } else {
                    -1
                }
            };
            *state = (*h, *l, *c, trend);
            Some(v * trend as f64)
        })
        .collect::<Vec<f64>>()
}

/// klinger volume oscillator
/// different from formula defined by https://www.investopedia.com/terms/k/klingeroscillator.asp
pub fn kvo(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: u8,
    long: u8,
) -> Vec<f64> {
    let vf = vforce(high, low, close, volume);
    let short_ma = smooth::ewma(&vf, short);
    let long_ma = smooth::ewma(&vf, long);
    short_ma[short_ma.len() - long_ma.len()..]
        .iter()
        .zip(long_ma)
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

/// quick stick
/// https://www.investopedia.com/terms/q/qstick.asp
pub fn qstick(open: &[f64], close: &[f64], window: u8) -> Vec<f64> {
    let q = close
        .iter()
        .zip(open.iter())
        .map(|(c, o)| c - o)
        .collect::<Vec<f64>>();
    smooth::ewma(&q, window)
}

fn wilder_sum(data: &[f64], window: u8) -> Vec<f64> {
    let initial = data[..(window - 1) as usize].iter().sum::<f64>();
    data[(window - 1) as usize..]
        .iter()
        .scan(initial, |state, x| {
            let ma = *state * (window - 1) as f64 / window as f64 + x;
            *state = ma;
            Some(ma)
        })
        .collect::<Vec<f64>>()
}

/// twiggs money flow
/// https://www.marketvolume.com/technicalanalysis/twiggsmoneyflow.asp
/// https://www.incrediblecharts.com/indicators/twiggs_money_flow.php
pub fn twiggs(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], window: u8) -> Vec<f64> {
    let data = izip!(&high[1..], &low[1..], &close[1..], &volume[1..]);
    // not using wilder moving average to minimise drift caused by floating point math
    let ad = wilder_sum(
        &data
            .scan(close[0], |state, (h, l, c, vol)| {
                let range_vol = vol
                    * ((2.0 * c - f64::min(*l, *state) - f64::max(*h, *state))
                        / (f64::max(*h, *state) - f64::min(*l, *state)));
                *state = *c;
                Some(range_vol)
            })
            .collect::<Vec<f64>>(),
        window,
    );
    ad.iter()
        .zip(wilder_sum(&volume[1..], window).iter())
        .map(|(range, vol)| range / vol)
        .collect()
}

/// shinohara intensity ratio
/// https://www.sevendata.co.jp/shihyou/technical/shinohara.html
pub fn shinohara(high: &[f64], low: &[f64], close: &[f64], period: u8) -> (Vec<f64>, Vec<f64>) {
    let high_win = high
        .windows(period.into())
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    let low_win = low
        .windows(period.into())
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    let close_win = close
        .windows(period.into())
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
    period: u8,
    smoothing: u8,
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
    let atr = smooth::wilder(&tr, period);
    let di_pos = izip!(smooth::wilder(&dm_pos, period), &atr)
        .map(|(di, tr)| di / tr * 100.0)
        .collect::<Vec<f64>>();
    let di_neg = izip!(smooth::wilder(&dm_neg, period), &atr)
        .map(|(di, tr)| di / tr * 100.0)
        .collect::<Vec<f64>>();
    let dx = izip!(&di_pos, &di_neg)
        .map(|(pos, neg)| f64::abs(pos - neg) / (pos + neg) * 100.0)
        .collect::<Vec<f64>>();
    (di_pos, di_neg, smooth::wilder(&dx, smoothing))
}

/// relative strength index
/// https://www.investopedia.com/terms/r/rsi.asp
pub fn rsi(data: &[f64], window: u8) -> Vec<f64> {
    let (gain, loss): (Vec<f64>, Vec<f64>) = data[1..]
        .iter()
        .zip(data[..data.len() - 1].iter())
        .map(|(curr, prev)| (f64::max(0.0, curr - prev), f64::min(0.0, curr - prev).abs()))
        .unzip();
    smooth::wilder(&gain, window)
        .iter()
        .zip(smooth::wilder(&loss, window).iter())
        .map(|(g, l)| 100.0 - (100.0 / (1.0 + (g / l))))
        .collect::<Vec<f64>>()
}

/// moving average convergence/divergence
/// https://www.investopedia.com/terms/m/macd.asp
pub fn macd(close: &[f64], short: u8, long: u8) -> Vec<f64> {
    let short_ma = smooth::ewma(close, short);
    let long_ma = smooth::ewma(close, long);
    short_ma[short_ma.len() - long_ma.len()..]
        .iter()
        .zip(long_ma)
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

/// chande momentum oscillator
/// https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
pub fn cmo(data: &[f64], window: u8) -> Vec<f64> {
    smooth::_cmo(data, window)
        .iter()
        .map(|x| x * 100.0)
        .collect::<Vec<f64>>()
}

/// centre of gravity
/// https://www.stockmaniacs.net/center-of-gravity-indicator/
pub fn cog(data: &[f64], window: u8) -> Vec<f64> {
    data.windows(window.into())
        .map(|w| {
            -w.iter()
                .rev()
                .enumerate()
                .map(|(i, e)| (e * (i + 1) as f64))
                .sum::<f64>()
                / w.iter().sum::<f64>()
        })
        .collect::<Vec<f64>>()
}

/// accumulation/distribution
/// https://www.investopedia.com/terms/a/accumulationdistribution.asp
pub fn ad(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    izip!(high, low, close, volume)
        .scan(0.0, |state, (h, l, c, vol)| {
            let mfm = ((c - l) - (h - c)) / (h - l);
            let mfv = mfm * vol;
            let adl = *state + mfv;
            *state = adl;
            Some(adl)
        })
        .collect::<Vec<f64>>()
}

/// accumulation/distribution
/// like yahoo
pub fn ad_yahoo(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    izip!(&high[1..], &low[1..], &close[1..], &volume[1..])
        .scan((close[0], 0.0), |state, (h, l, c, vol)| {
            let mfm = if *c > state.0 {
                c - f64::min(*l, state.0)
            } else {
                c - f64::max(*h, state.0)
            };
            let mfv = mfm * vol;
            let adl = state.1 + mfv;
            *state = (*c, adl);
            Some(adl)
        })
        .collect::<Vec<f64>>()
}

/// elder ray
/// https://www.investopedia.com/articles/trading/03/022603.asp
/// returns tuple of bull power vec and bear power vec
pub fn elder_ray(high: &[f64], low: &[f64], close: &[f64], window: u8) -> (Vec<f64>, Vec<f64>) {
    let close_ma = smooth::ewma(close, window);
    izip!(
        &high[high.len() - close_ma.len()..],
        &low[low.len() - close_ma.len()..],
        close_ma
    )
    .map(|(h, l, c)| (h - c, l - c))
    .unzip()
}

/// elder force index
/// https://www.investopedia.com/articles/trading/03/031203.asp
pub fn elder_force(close: &[f64], volume: &[f64], window: u8) -> Vec<f64> {
    smooth::ewma(
        &izip!(&close[..close.len() - 1], &close[1..], &volume[1..])
            .map(|(prev, curr, vol)| (curr - prev) * vol)
            .collect::<Vec<f64>>(),
        window,
    )
}

/// williams alligator
/// https://www.investopedia.com/articles/trading/072115/exploring-williams-alligator-indicator.asp
pub fn alligator(_data: &[f64]) {}

/// money flow index
/// https://www.investopedia.com/terms/m/mfi.asp
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], window: u8) -> Vec<f64> {
    let (pos_mf, neg_mf): (Vec<f64>, Vec<f64>) =
        izip!(&high[1..], &low[1..], &close[1..], &volume[1..])
            .scan(
                (high[0] + low[0] + close[0]) / 3.0,
                |state, (h, l, c, vol)| {
                    let hlc = (h + l + c) / 3.0;
                    let pos_mf = if hlc > *state { hlc * vol } else { 0.0 };
                    let neg_mf = if hlc < *state { hlc * vol } else { 0.0 };
                    *state = hlc;
                    Some((pos_mf, neg_mf))
                },
            )
            .unzip();
    pos_mf
        .windows(window.into())
        .zip(neg_mf.windows(window.into()))
        .map(|(pos, neg)| {
            100.0 - (100.0 / (1.0 + pos.iter().sum::<f64>() / neg.iter().sum::<f64>()))
        })
        .collect::<Vec<f64>>()
}

/// chaikin money flow
/// https://corporatefinanceinstitute.com/resources/equities/chaikin-money-flow-cmf/
pub fn cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], window: u8) -> Vec<f64> {
    izip!(high, low, close, volume)
        .map(|(h, l, c, vol)| vol * ((c - l) - (h - c)) / (h - l))
        .collect::<Vec<f64>>()
        .windows(window.into())
        .zip(volume.windows(window.into()))
        .map(|(mfv_win, v_win)| mfv_win.iter().sum::<f64>() / v_win.iter().sum::<f64>())
        .collect::<Vec<f64>>()
}

/// chaikin volatility
/// https://www.tradingview.com/chart/AUDUSD/gjfxqWqW-What-Is-a-Chaikin-Volatility-Indicator-in-Trading/
/// https://theforexgeek.com/chaikins-volatility-indicator/
pub fn cvi(high: &[f64], low: &[f64], window: u8, rate_of_change: u8) -> Vec<f64> {
    smooth::ewma(
        &high
            .iter()
            .zip(low)
            .map(|(h, l)| h - l)
            .collect::<Vec<f64>>(),
        window,
    )
    .windows((rate_of_change + 1).into())
    .map(|w| 100.0 * (w.last().unwrap() / w.first().unwrap() - 1.0))
    .collect::<Vec<f64>>()
}

/// Williams Percent Range
/// https://www.investopedia.com/terms/w/williamsr.asp
pub fn wpr(high: &[f64], low: &[f64], close: &[f64], window: u8) -> Vec<f64> {
    izip!(
        high.windows(window.into()),
        low.windows(window.into()),
        &close[(window - 1).into()..]
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
pub fn vortex(high: &[f64], low: &[f64], close: &[f64], window: u8) -> (Vec<f64>, Vec<f64>) {
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
    .windows(window.into())
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

/// percent oscillator
/// pass in any data (close, high, low, etc...), and two window ranges
pub fn po(data: &[f64], short: u8, long: u8) -> Vec<f64> {
    let short_ma = smooth::ewma(data, short);
    let long_ma = smooth::ewma(data, long);
    short_ma[short_ma.len() - long_ma.len()..]
        .iter()
        .zip(long_ma)
        .map(|(x, y)| 100.0 * (x / y - 1.0))
        .collect::<Vec<f64>>()
}

/// vertical horizontal filter
/// https://www.upcomingtrader.com/blog/the-vertical-horizontal-filter-a-traders-guide-to-market-phases/
pub fn vhf(high: &[f64], low: &[f64], close: &[f64], window: u8) -> Vec<f64> {
    let diffs = &close[1..]
        .iter()
        .zip(&close[..close.len() - 1])
        .map(|(curr, prev)| (curr - prev).abs())
        .collect::<Vec<f64>>();
    izip!(
        diffs.windows(window.into()),
        high.windows(window.into()).skip(1),
        low.windows(window.into()).skip(1)
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
    win1: u8,
    win2: u8,
    win3: u8,
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
        .windows(win3.into())
        .map(|w| {
            let (bp_sum1, tr_sum1) = w
                .iter()
                .skip((win3 - win1).into())
                .fold((0.0, 0.0), |acc, (bp, tr)| (acc.0 + bp, acc.1 + tr));
            let (bp_sum2, tr_sum2) = w
                .iter()
                .skip((win3 - win2).into())
                .fold((0.0, 0.0), |acc, (bp, tr)| (acc.0 + bp, acc.1 + tr));
            let (bp_sum3, tr_sum3) = w
                .iter()
                .fold((0.0, 0.0), |acc, (bp, tr)| (acc.0 + bp, acc.1 + tr));
            100.0 * (bp_sum1 / tr_sum1 * 4.0 + bp_sum2 / tr_sum2 * 2.0 + bp_sum3 / tr_sum3)
                / (4 + 2 + 1) as f64
        })
        .collect::<Vec<f64>>()
}

/// pretty good oscillator
/// https://library.tradingtechnologies.com/trade/chrt-ti-pretty-good-oscillator.html
pub fn pgo(high: &[f64], low: &[f64], close: &[f64], window: u8) -> Vec<f64> {
    let atr = smooth::ewma(&_true_range(high, low, close).collect::<Vec<f64>>(), window);
    let sma_close = smooth::sma(close, window);
    izip!(
        &close[close.len() - atr.len()..],
        &sma_close[sma_close.len() - atr.len()..],
        atr
    )
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
pub fn ulcer(data: &[f64], window: u8) -> Vec<f64> {
    let highest = data
        .windows(window.into())
        .map(|w| w.iter().fold(f64::NAN, |state, &x| state.max(x)))
        .collect::<Vec<f64>>();
    smooth::sma(
        &highest
            .iter()
            .zip(&data[data.len() - highest.len()..])
            .map(|(high, c)| (100.0 * (c - high) / high).powi(2))
            .collect::<Vec<f64>>(),
        window,
    )
    .iter()
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
pub fn hlc3(high: &[f64], low: &[f64], close: &[f64], window: u8) -> Vec<f64> {
    smooth::sma(
        &izip!(high, low, close)
            .map(|(h, l, c)| (h + l + c) / 3.0)
            .collect::<Vec<f64>>(),
        window,
    )
}

/// Triple Exponential Average
/// https://www.investopedia.com/terms/t/trix.asp
pub fn trix(close: &[f64], window: u8) -> Vec<f64> {
    let ema3 = smooth::ewma(&smooth::ewma(&smooth::ewma(close, window), window), window);
    ema3[..ema3.len() - 1]
        .iter()
        .zip(&ema3[1..])
        .map(|(prev, curr)| 100.0 * (curr - prev) / prev)
        .collect::<Vec<f64>>()
}

/// trend intensity index
/// https://www.marketvolume.com/technicalanalysis/trendintensityindex.asp
pub fn tii(data: &[f64], window: u8) -> Vec<f64> {
    smooth::sma(data, window)
        .iter()
        .zip(&data[(window - 1) as usize..])
        .map(|(avg, actual)| {
            let dev: f64 = actual - avg;
            let pos_dev = if dev > 0.0 { dev } else { 0.0 };
            let neg_dev = if dev < 0.0 { dev.abs() } else { 0.0 };
            (pos_dev, neg_dev)
        })
        .collect::<Vec<(f64, f64)>>()
        .windows(u8::div_ceil(window, 2).into())
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

/// trade volume index
/// https://www.investopedia.com/terms/t/tradevolumeindex.asp
pub fn tvi(close: &[f64], volume: &[f64], min_tick: f64) -> Vec<f64> {
    izip!(&close[..close.len() - 1], &close[1..], &volume[1..],)
        .scan((1, 0.0), |state, (prev, curr, vol)| {
            let direction = if curr - prev > min_tick {
                1
            } else if prev - curr > min_tick {
                -1
            } else {
                state.0
            };
            let tvi = state.1 + direction as f64 * vol;
            *state = (direction, tvi);
            Some(tvi)
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
    window: u8,
    multiplier: f64,
) -> Vec<f64> {
    // TODO: needs a test for when it actually flips to use upper band line
    let atr = smooth::wilder(&_true_range(high, low, close).collect::<Vec<f64>>(), window);
    izip!(
        &high[window.into()..],
        &low[window.into()..],
        &close[window.into()..],
        &atr
    )
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
pub fn stochastic(high: &[f64], low: &[f64], close: &[f64], window: u8) -> (Vec<f64>, Vec<f64>) {
    let fast_k = smooth::sma(
        &izip!(
            high.windows(window.into()),
            low.windows(window.into()),
            &close[(window - 1).into()..]
        )
        .map(|(h, l, c)| {
            let hh = h.iter().fold(f64::NAN, |state, &x| state.max(x));
            let ll = l.iter().fold(f64::NAN, |state, &x| state.min(x));
            100.0 * (c - ll) / (hh - ll)
        })
        .collect::<Vec<f64>>(),
        3,
    );
    let k = smooth::sma(&fast_k, 3);
    (fast_k, k)
}

fn _stc(series: &[f64], window: u8) -> Vec<f64> {
    smooth::wilder(
        &series
            .windows(window.into())
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
}

/// Shaff Trend Cycle
/// https://www.investopedia.com/articles/forex/10/schaff-trend-cycle-indicator.asp
/// https://www.stockmaniacs.net/schaff-trend-cycle-indicator/
pub fn stc(close: &[f64], window: u8, short: u8, long: u8) -> Vec<f64> {
    let series = macd(close, short, long);
    _stc(&_stc(&series, window), window)
}
