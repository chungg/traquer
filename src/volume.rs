use itertools::izip;

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
    short: usize,
    long: usize,
) -> Vec<f64> {
    let vf = vforce(high, low, close, volume);
    let short_ma = smooth::ewma(&vf, short);
    let long_ma = smooth::ewma(&vf, long);
    short_ma
        .skip(long - short)
        .zip(long_ma)
        .map(|(x, y)| x - y)
        .collect::<Vec<f64>>()
}

fn wilder_sum(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let initial = data[..(window - 1)].iter().sum::<f64>();
    data[(window - 1)..].iter().scan(initial, move |state, x| {
        let ma = *state * (window - 1) as f64 / window as f64 + x;
        *state = ma;
        Some(ma)
    })
}

/// twiggs money flow
/// https://www.marketvolume.com/technicalanalysis/twiggsmoneyflow.asp
/// https://www.incrediblecharts.com/indicators/twiggs_money_flow.php
pub fn twiggs(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
    let data = izip!(&high[1..], &low[1..], &close[1..], &volume[1..]);
    // not using wilder moving average to minimise drift caused by floating point math
    wilder_sum(
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
    )
    .zip(wilder_sum(&volume[1..], window))
    .map(|(range, vol)| range / vol)
    .collect()
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

/// elder force index
/// https://www.investopedia.com/articles/trading/03/031203.asp
pub fn elder_force(close: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
    smooth::ewma(
        &izip!(&close[..close.len() - 1], &close[1..], &volume[1..])
            .map(|(prev, curr, vol)| (curr - prev) * vol)
            .collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>()
}

/// money flow index
/// https://www.investopedia.com/terms/m/mfi.asp
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
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
        .windows(window)
        .zip(neg_mf.windows(window))
        .map(|(pos, neg)| {
            100.0 - (100.0 / (1.0 + pos.iter().sum::<f64>() / neg.iter().sum::<f64>()))
        })
        .collect::<Vec<f64>>()
}

/// chaikin money flow
/// https://corporatefinanceinstitute.com/resources/equities/chaikin-money-flow-cmf/
pub fn cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
    izip!(high, low, close, volume)
        .map(|(h, l, c, vol)| vol * ((c - l) - (h - c)) / (h - l))
        .collect::<Vec<f64>>()
        .windows(window)
        .zip(volume.windows(window))
        .map(|(mfv_win, v_win)| mfv_win.iter().sum::<f64>() / v_win.iter().sum::<f64>())
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

/// Ease of Movement
/// https://www.investopedia.com/terms/e/easeofmovement.asp
pub fn ease(high: &[f64], low: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
    smooth::sma(
        &(1..high.len())
            .map(|i| {
                (high[i] + low[i] - high[i - 1] - low[i - 1])
                    / 2.0
                    / (volume[i] / 100000000.0 / (high[i] - low[i]))
            })
            .collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>()
}

/// On-Balance Volume
/// https://www.investopedia.com/terms/o/onbalancevolume.asp
pub fn obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    close
        .windows(2)
        .enumerate()
        .scan(0.0, |state, (i, pairs)| {
            *state += (pairs[1] - pairs[0]).signum() * volume[i + 1];
            Some(*state)
        })
        .collect::<Vec<f64>>()
}

/// Market Facilitation Index
/// https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/market_facilitation
pub fn bw_mfi(high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low)
        .zip(volume)
        .map(|((h, l), vol)| (h - l) / vol * (10.0_f64).powi(6))
        .collect::<Vec<f64>>()
}
