use itertools::{izip, multiunzip};

use crate::smooth;

fn vforce(h: &Vec<f64>, l: &Vec<f64>, c: &Vec<f64>, v: &Vec<f64>) -> Vec<f64> {
    izip!(&h[1..], &l[1..], &c[1..], &v[1..])
        .scan(
            (h[0], l[0], c[0], 99, 0.0, h[0] - l[0]),
            |state, (h, l, c, v)| {
                let trend: i8 = {
                    if h + l + c > state.0 + state.1 + state.2 {
                        1
                    } else {
                        -1
                    }
                };
                let dm: f64 = h - l;
                let cm: f64 = {
                    if trend == state.3 {
                        state.4 + dm
                    } else {
                        state.5 + dm
                    }
                };
                *state = (*h, *l, *c, trend, cm, dm);
                Some(v * 2.0 * ((dm / cm) - 1.0) * trend as f64 * 100.0)
            },
        )
        .collect::<Vec<f64>>()
}

///  klinger oscillator
///  https://www.investopedia.com/terms/k/klingeroscillator.asp
pub fn klinger(
    h: &Vec<f64>,
    l: &Vec<f64>,
    c: &Vec<f64>,
    v: &Vec<f64>,
    short: u8,
    long: u8,
) -> Vec<f64> {
    let vf = vforce(h, l, c, v);
    smooth::ewma(&vf, short)
        .iter()
        .zip(smooth::ewma(&vf, long).iter())
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

fn vforce_simple(h: &Vec<f64>, l: &Vec<f64>, c: &Vec<f64>, v: &Vec<f64>) -> Vec<f64> {
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
/// designed to match yahoo
pub fn klinger_vol(
    h: &Vec<f64>,
    l: &Vec<f64>,
    c: &Vec<f64>,
    v: &Vec<f64>,
    short: u8,
    long: u8,
) -> Vec<f64> {
    let vf = vforce_simple(h, l, c, v);
    smooth::ewma(&vf, short)
        .iter()
        .zip(smooth::ewma(&vf, long).iter())
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

/// quick stick
/// https://www.investopedia.com/terms/q/qstick.asp
pub fn qstick(o: &Vec<f64>, c: &Vec<f64>, window: u8) -> Vec<f64> {
    let q = c
        .iter()
        .zip(o.iter())
        .map(|(close, open)| close - open)
        .collect::<Vec<f64>>();
    smooth::ewma(&q, window)
}

/// twiggs money flow
/// https://www.marketvolume.com/technicalanalysis/twiggsmoneyflow.asp
/// https://www.incrediblecharts.com/indicators/twiggs_money_flow.php
pub fn twiggs(h: &Vec<f64>, l: &Vec<f64>, c: &Vec<f64>, v: &Vec<f64>, window: u8) -> Vec<f64> {
    let data = izip!(h, l, c, v);
    let ewma_range = smooth::wilder(
        &data
            .scan(f64::NAN, |state, (high, low, close, vol)| {
                let range_vol = vol
                    * ((close - f64::min(*low, *state))
                        / (f64::max(*high, *state) - f64::min(*low, *state))
                        * 2.0
                        - 1.0);
                *state = *close;
                Some(range_vol)
            })
            .collect::<Vec<f64>>(),
        window,
    );
    ewma_range
        .iter()
        .zip(smooth::wilder(v, window).iter())
        .map(|(range, vol)| range / vol)
        .collect()
}

/// shinohara intensity ratio
/// https://www.sevendata.co.jp/shihyou/technical/shinohara.html
pub fn shinohara(h: &Vec<f64>, l: &Vec<f64>, c: &Vec<f64>, period: u8) -> (Vec<f64>, Vec<f64>) {
    // yahoo uses close rather than open for weak ratio described above
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
    h: &Vec<f64>,
    l: &Vec<f64>,
    c: &Vec<f64>,
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
pub fn rsi(values: &Vec<f64>, window: u8) -> Vec<f64> {
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
pub fn macd(close: &Vec<f64>, fast: u8, slow: u8) -> Vec<f64> {
    smooth::ewma(close, fast)
        .iter()
        .zip(smooth::ewma(close, slow))
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

/// chande momentum oscillator
/// https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
pub fn cmo(data: &Vec<f64>, window: u8) -> Vec<f64> {
    smooth::_cmo(data, window)
        .iter()
        .map(|x| x * 100.0)
        .collect::<Vec<f64>>()
}

// centre of gravity
// https://www.stockmaniacs.net/center-of-gravity-indicator/
pub fn cog(data: &Vec<f64>, window: u8) -> Vec<f64> {
    data.windows(window.into())
        .map(|w| {
            let series_sum: f64 = w.iter().sum();
            -w.iter()
                .rev()
                .enumerate()
                .map(|(i, e)| (e * (i + 1) as f64))
                .sum::<f64>()
                / w.iter().sum::<f64>()
        })
        .collect::<Vec<f64>>()
}
