use itertools::{izip, multiunzip};

use crate::smooth;

fn vforce(h: &[f64], l: &[f64], c: &[f64], v: &[f64]) -> Vec<f64> {
    izip!(&h[1..], &l[1..], &c[1..], &v[1..])
        .scan((h[0], l[0], c[0], 99), |state, (h, l, c, v)| {
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
pub fn kvo(h: &[f64], l: &[f64], c: &[f64], v: &[f64], short: u8, long: u8) -> Vec<f64> {
    let vf = vforce(h, l, c, v);
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
pub fn qstick(o: &[f64], c: &[f64], window: u8) -> Vec<f64> {
    let q = c
        .iter()
        .zip(o.iter())
        .map(|(close, open)| close - open)
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
pub fn twiggs(h: &[f64], l: &[f64], c: &[f64], v: &[f64], window: u8) -> Vec<f64> {
    let data = izip!(&h[1..], &l[1..], &c[1..], &v[1..]);
    // not using wilder moving average to minimise drift caused by floating point math
    let ad = wilder_sum(
        &data
            .scan(c[0], |state, (high, low, close, vol)| {
                let range_vol = vol
                    * ((2.0 * close - f64::min(*low, *state) - f64::max(*high, *state))
                        / (f64::max(*high, *state) - f64::min(*low, *state)));
                *state = *close;
                Some(range_vol)
            })
            .collect::<Vec<f64>>(),
        window,
    );
    ad.iter()
        .zip(wilder_sum(&v[1..], window).iter())
        .map(|(range, vol)| range / vol)
        .collect()
}

/// shinohara intensity ratio
/// https://www.sevendata.co.jp/shihyou/technical/shinohara.html
pub fn shinohara(h: &[f64], l: &[f64], c: &[f64], period: u8) -> (Vec<f64>, Vec<f64>) {
    let high = h
        .windows(period.into())
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    let low = l
        .windows(period.into())
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    let close = c
        .windows(period.into())
        .map(|w| w.iter().sum())
        .collect::<Vec<f64>>();
    // yahoo uses close rather than open for weak ratio described above
    let weak_ratio = izip!(&high, &low, &close)
        .map(|(h, l, c)| 100.0 * (h - c) / (c - l))
        .collect::<Vec<f64>>();
    let strong_ratio = izip!(&high[1..], &low[1..], &close[..close.len() - 1])
        .map(|(h, l, c)| 100.0 * (h - c) / (c - l))
        .collect::<Vec<f64>>();
    (strong_ratio, weak_ratio)
}

/// average directional index
/// https://www.investopedia.com/terms/a/adx.asp
pub fn adx(
    h: &[f64],
    l: &[f64],
    c: &[f64],
    period: u8,
    smoothing: u8,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (dm_pos, dm_neg, tr): (Vec<_>, Vec<_>, Vec<_>) = multiunzip(
        izip!(
            &h[..h.len() - 1],
            &h[1..],
            &l[..l.len() - 1],
            &l[1..],
            &c[..c.len() - 1],
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
pub fn rsi(values: &[f64], window: u8) -> Vec<f64> {
    let (gain, loss): (Vec<f64>, Vec<f64>) = values[1..]
        .iter()
        .zip(values[..values.len() - 1].iter())
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
pub fn macd(close: &[f64], fast: u8, slow: u8) -> Vec<f64> {
    let fast_ma = smooth::ewma(close, fast);
    let slow_ma = smooth::ewma(close, slow);
    fast_ma[fast_ma.len() - slow_ma.len()..]
        .iter()
        .zip(slow_ma)
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
pub fn ad(h: &[f64], l: &[f64], c: &[f64], v: &[f64]) -> Vec<f64> {
    izip!(h, l, c, v)
        .scan(0.0, |state, (high, low, close, vol)| {
            let mfm = ((close - low) - (high - close)) / (high - low);
            let mfv = mfm * vol;
            let adl = *state + mfv;
            *state = adl;
            Some(adl)
        })
        .collect::<Vec<f64>>()
}

/// accumulation/distribution
/// like yahoo
pub fn ad_yahoo(h: &[f64], l: &[f64], c: &[f64], v: &[f64]) -> Vec<f64> {
    izip!(&h[1..], &l[1..], &c[1..], &v[1..])
        .scan((c[0], 0.0), |state, (high, low, close, vol)| {
            let mfm = if *close > state.0 {
                close - f64::min(*low, state.0)
            } else {
                close - f64::max(*high, state.0)
            };
            let mfv = mfm * vol;
            let adl = state.1 + mfv;
            *state = (*close, adl);
            Some(adl)
        })
        .collect::<Vec<f64>>()
}

/// elder ray
/// https://www.investopedia.com/articles/trading/03/022603.asp
/// returns tuple of bull power vec and bear power vec
pub fn elder_ray(h: &[f64], l: &[f64], c: &[f64], window: u8) -> (Vec<f64>, Vec<f64>) {
    let close_ma = smooth::ewma(c, window);
    izip!(
        &h[h.len() - close_ma.len()..],
        &l[l.len() - close_ma.len()..],
        close_ma
    )
    .map(|(high, low, close)| (high - close, low - close))
    .unzip()
}

/// elder force index
/// https://www.investopedia.com/articles/trading/03/031203.asp
pub fn elder_force(c: &[f64], v: &[f64], window: u8) -> Vec<f64> {
    smooth::ewma(
        &izip!(&c[..c.len() - 1], &c[1..], &v[1..])
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
pub fn mfi(h: &[f64], l: &[f64], c: &[f64], v: &[f64], window: u8) -> Vec<f64> {
    let (pos_mf, neg_mf): (Vec<f64>, Vec<f64>) = izip!(&h[1..], &l[1..], &c[1..], &v[1..])
        .scan(
            (h[0] + l[0] + c[0]) / 3.0,
            |state, (high, low, close, vol)| {
                let hlc = (high + low + close) / 3.0;
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
