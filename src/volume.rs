//! Volume Indicators
//!
//! Provides technical indicators that measures the efficiency of price movement
//! by analyzing the relationship between price changes and trading volume.
//! Depending on the indicator, it may be a momentum indicator or trend indicator.
use std::iter;

use itertools::izip;

use crate::smooth;

fn vforce<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
) -> impl Iterator<Item = f64> + 'a {
    izip!(&high[1..], &low[1..], &close[1..], &volume[1..]).scan(
        (high[0], low[0], close[0], 0.0),
        |state, (h, l, c, v)| {
            let trend = ((h + l + c) - (state.0 + state.1 + state.2)).signum();
            *state = (*h, *l, *c, trend);
            Some(v * trend)
        },
    )
}

/// Klinger Volume Oscillator (KVO)
///
/// Developed by Stephen Klinger. It helps determine the long-term trend of money flow
/// while remaining sensitive enough to detect short-term fluctuations.
///
/// Note: This is different from formula defined in source. The vforce value is simply
/// volume * trend
///
/// # Usage
///
/// When the value is above its signal line and/or it crosses above 0, it suggests an uptrend.
///
/// # Source
/// https://www.investopedia.com/terms/k/klingeroscillator.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::kvo(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     3, 6).collect::<Vec<f64>>();
///
/// ```
pub fn kvo<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    short: usize,
    long: usize,
) -> impl Iterator<Item = f64> + 'a {
    let vf = vforce(high, low, close, volume).collect::<Vec<f64>>();
    let short_ma = smooth::ewma(&vf, short);
    let long_ma = smooth::ewma(&vf, long);
    iter::once(f64::NAN).chain(
        short_ma
            .zip(long_ma)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    )
}

fn wilder_sum(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let initial = data[..(window - 1)].iter().sum::<f64>();
    data[(window - 1)..].iter().scan(initial, move |state, x| {
        let ma = *state * (window - 1) as f64 / window as f64 + x;
        *state = ma;
        Some(ma)
    })
}

/// Twiggs Money Flow
///
/// Developed by Colin Twiggs that measures the flow of money into and out of a security.
/// It's similar to the Accumulation/Distribution Line. A rising TMF indicates buying pressure,
/// as more money is flowing into the security.
///
/// # Usage
///
/// A value above 0 suggests an uptrend.
///
/// # Source
///
/// https://www.marketvolume.com/technicalanalysis/twiggsmoneyflow.asp
/// https://www.incrediblecharts.com/indicators/twiggs_money_flow.php
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::twiggs(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn twiggs<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    let data = izip!(&high[1..], &low[1..], &close[1..], &volume[1..]);
    // not using wilder moving average to minimise drift caused by floating point math
    iter::repeat(f64::NAN).take(window).chain(
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
        .collect::<Vec<f64>>(),
    )
}

/// Accumulation/Distribution (A/D) indicator
///
/// Developed by Marc Chaikin. A momentum indicator that measures the flow of money into
/// and out of a security.
///
/// Calculated by multiplying the money flow multiplier (which is based on the security's
/// price and volume) by the money flow volume (which is the volume at the current price level).
/// This function supports alternate logic to consider prior close like yahoo
///
/// # Usage
///
/// An increasing value suggests an uptrend.
///
/// # Source
///
/// https://www.investopedia.com/terms/a/accumulationdistribution.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::ad(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     None).collect::<Vec<f64>>();
///
/// ```
pub fn ad<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    alt: Option<bool>,
) -> Box<dyn Iterator<Item = f64> + 'a> {
    if !alt.unwrap_or(false) {
        Box::new(
            izip!(high, low, close, volume).scan(0.0, |state, (h, l, c, vol)| {
                let mfm = ((c - l) - (h - c)) / (h - l);
                let mfv = mfm * vol;
                let adl = *state + mfv;
                *state = adl;
                Some(adl)
            }),
        )
    } else {
        // alternate logic to consider prior close like yahoo
        Box::new(iter::once(f64::NAN).chain(
            izip!(&high[1..], &low[1..], &close[1..], &volume[1..]).scan(
                (close[0], 0.0),
                |state, (h, l, c, vol)| {
                    let mfm = if *c > state.0 {
                        c - f64::min(*l, state.0)
                    } else {
                        c - f64::max(*h, state.0)
                    };
                    let mfv = mfm * vol;
                    let adl = state.1 + mfv;
                    *state = (*c, adl);
                    Some(adl)
                },
            ),
        ))
    }
}

/// Elder Force Index
///
/// Calculated by multiplying the change in price by the volume traded during that period.
/// A high EFI value indicates a strong price move with high volume, which can be a sign of
/// a strong trend
///
/// # Usage
///
/// A value above 0 suggests an uptrend.
///
/// # Source
///
/// https://www.investopedia.com/articles/trading/03/031203.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::elder_force(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn elder_force<'a>(
    close: &'a [f64],
    volume: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(
        smooth::ewma(
            &izip!(&close[..close.len() - 1], &close[1..], &volume[1..])
                .map(|(prev, curr, vol)| (curr - prev) * vol)
                .collect::<Vec<f64>>(),
            window,
        )
        .collect::<Vec<f64>>(),
    )
}

/// Money Flow Index
///
/// Calculated by using the typical price and the volume traded during that period.
/// A high MFI value (above 80) indicates that the security is overbought, and a
/// correction may be due.
///
/// # Usage
///
/// Typically, a value above 80 suggests overbought and a value below 20, oversold.
///
/// # Source
///
/// https://www.investopedia.com/terms/m/mfi.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::mfi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn mfi<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
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
    iter::repeat(f64::NAN).take(window).chain(
        pos_mf
            .windows(window)
            .zip(neg_mf.windows(window))
            .map(|(pos, neg)| {
                100.0 - (100.0 / (1.0 + pos.iter().sum::<f64>() / neg.iter().sum::<f64>()))
            })
            .collect::<Vec<f64>>(),
    )
}

/// Chaikin Money Flow
///
/// Calculated by multiplying the money flow multiplier (which is based on the
/// security's price and volume) by the money flow volume (which is the volume at
/// the current price level). A positive CMF value indicates that money is flowing into
/// the security, which can be a sign of buying pressure.
///
/// # Usage
///
/// A value above 0 suggests an uptrend.
///
/// # Source
///
/// https://corporatefinanceinstitute.com/resources/equities/chaikin-money-flow-cmf/
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::cmf(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn cmf<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        izip!(high, low, close, volume)
            .map(|(h, l, c, vol)| vol * ((c - l) - (h - c)) / (h - l))
            .collect::<Vec<f64>>()
            .windows(window)
            .zip(volume.windows(window))
            .map(|(mfv_win, v_win)| mfv_win.iter().sum::<f64>() / v_win.iter().sum::<f64>())
            .collect::<Vec<f64>>(),
    )
}

/// Trade Volume Index
///
/// Measures the flow of money into and out of a security by analyzing the trading volume at
/// different price levels.
///
/// # Usage
///
/// An increasing value suggests an uptrend.
///
/// # Source
///
/// https://www.investopedia.com/terms/t/tradevolumeindex.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::tvi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     0.5).collect::<Vec<f64>>();
///
/// ```
pub fn tvi<'a>(
    close: &'a [f64],
    volume: &'a [f64],
    min_tick: f64,
) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(
        izip!(&close[..close.len() - 1], &close[1..], &volume[1..],).scan(
            (1, 0.0),
            move |state, (prev, curr, vol)| {
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
            },
        ),
    )
}

/// Ease of Movement
///
/// Ease shows the amount of volume required to move prices by a certain amount.
/// A high Ease value indicates that prices can move easily with low volume, while a
/// low Ease value indicates that prices are difficult to move and require high volume.
///
/// # Usage
///
/// A value above 0 suggests an uptrend.
///
/// # Source
/// https://www.investopedia.com/terms/e/easeofmovement.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::ease(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     6).collect::<Vec<f64>>();
///
/// ```
pub fn ease<'a>(
    high: &'a [f64],
    low: &'a [f64],
    volume: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(
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
        .collect::<Vec<f64>>(),
    )
}

/// On-Balance Volume
///
/// Shows the cumulative total of volume traded on up days minus the cumulative total of
/// volume traded on down days.
///
/// # Usage
///
/// An increasing value suggests an uptrend.
///
/// # Source
///
/// https://www.investopedia.com/terms/o/onbalancevolume.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::obv(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0]).collect::<Vec<f64>>();
///
/// ```
pub fn obv<'a>(close: &'a [f64], volume: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(close.windows(2).enumerate().scan(0.0, |state, (i, pairs)| {
        *state += (pairs[1] - pairs[0]).signum() * volume[i + 1];
        Some(*state)
    }))
}

/// Market Facilitation Index
///
/// Shows the amount of price change per unit of volume traded. A high BW MFI value
/// indicates that prices are moving efficiently with low volume, while a low BW MFI
/// value indicates that prices are moving inefficiently with high volume.
///
/// # Usage
///
/// If both value and volume increases, suggests uptrend.
///
/// # Source
///
/// https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/market_facilitation
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::bw_mfi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0]).collect::<Vec<f64>>();
///
/// ```
pub fn bw_mfi<'a>(
    high: &'a [f64],
    low: &'a [f64],
    volume: &'a [f64],
) -> impl Iterator<Item = f64> + 'a {
    high.iter()
        .zip(low)
        .zip(volume)
        .map(|((h, l), vol)| (h - l) / vol * (10.0_f64).powi(6))
}

/// Positive Volume Index
///
/// Based on price moves depending on whether the current volume is higher than
/// the previous period.
///
/// # Usage
///
/// When above the one year average, confirmation of uptrend.
///
/// # Source
///
/// https://www.investopedia.com/terms/p/pvi.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::pvi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0]).collect::<Vec<f64>>();
///
/// ```
pub fn pvi<'a>(data: &'a [f64], volume: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(data.windows(2).zip(volume.windows(2)).scan(
        100.0,
        |state, (c, vol)| {
            if vol[1] > vol[0] {
                *state *= c[1] / c[0];
            }
            Some(*state)
        },
    ))
}

/// Negative Volume Index
///
/// Based on price moves depending on whether the current volume is higher than
/// the previous period.
///
/// # Usage
///
/// When above the one year average, confirmation of downtrend.
///
/// # Source
///
/// https://www.investopedia.com/terms/n/nvi.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::nvi(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0]).collect::<Vec<f64>>();
///
/// ```
pub fn nvi<'a>(data: &'a [f64], volume: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(data.windows(2).zip(volume.windows(2)).scan(
        100.0,
        |state, (c, vol)| {
            if vol[1] < vol[0] {
                *state *= c[1] / c[0];
            }
            Some(*state)
        },
    ))
}

/// Volume Weighted Average Price (VWAP)
///
/// Measures the average typical price by volume. Tracks similar to a moving average.
///
/// # Usage
///
/// Designed for intraday data, instruments with prices below VWAP may be considered undervalued.
///
/// # Source
///
/// https://www.investopedia.com/terms/v/vwap.asp
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::vwap(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0], None).collect::<Vec<f64>>();
///
/// ```
pub fn vwap<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    reset_idx: Option<&'a [usize]>,
) -> impl Iterator<Item = f64> + 'a {
    // NOTE: assumes reset_idx is sorted
    let mut reset_idx = reset_idx.unwrap_or(&[close.len()]).to_vec();
    izip!(high, low, close, volume).enumerate().scan(
        (0.0, 0.0),
        move |state, (idx, (h, l, c, vol))| {
            let (mut tpv_sum, mut vol_sum) = state;
            if idx == reset_idx[0] {
                tpv_sum = 0.0;
                vol_sum = 0.0;
                reset_idx.rotate_left(1);
            }
            tpv_sum += (h + l + c) / 3.0 * vol;
            vol_sum += vol;
            *state = (tpv_sum, vol_sum);
            Some(tpv_sum / vol_sum)
        },
    )
}

/// Volume Weighted Moving Average
///
/// Measures price by volume. Tracks similar to a moving average. A period with a
/// higher volume will significantly influence the value more than a period with a lower volume.
///
/// # Usage
///
/// Show trend like any normal moving average.
///
/// # Source
///
/// https://howtotrade.com/indicators/volume-weighted-moving-average/
///
/// # Examples
///
/// ```
/// use traquer::volume;
///
/// volume::vwma(
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0],
///     &vec![1.0,2.0,3.0,4.0,5.0,6.0,4.0,5.0], 3).collect::<Vec<f64>>();
///
/// ```
pub fn vwma<'a>(
    data: &'a [f64],
    volume: &'a [f64],
    window: usize,
) -> impl Iterator<Item = f64> + 'a {
    iter::repeat(f64::NAN).take(window - 1).chain(
        data.windows(window)
            .zip(volume.windows(window))
            .map(|(data_w, vol_w)| {
                data_w
                    .iter()
                    .zip(vol_w)
                    .fold(0.0, |acc, (x, v)| acc + x * v)
                    / vol_w.iter().sum::<f64>()
            }),
    )
}
