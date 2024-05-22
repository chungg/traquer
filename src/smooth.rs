use itertools::izip;
use std::iter;

// exponentially weighted moving average
pub fn ewma(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let initial = data[..window].iter().sum::<f64>() / window as f64;
    let alpha = 2.0 / (window + 1) as f64;
    iter::once(initial).chain(data[window..].iter().scan(initial, move |state, &x| {
        *state = x * alpha + *state * (1.0 - alpha);
        Some(*state)
    }))
}

/// simple moving average
pub fn sma(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.windows(window)
        .map(move |w| w.iter().sum::<f64>() / window as f64)
}

/// double exponential moving average
/// https://www.investopedia.com/terms/d/double-exponential-moving-average.asp
pub fn dema(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let ma = ewma(data, window).collect::<Vec<f64>>();
    let mama = ewma(&ma, window).collect::<Vec<f64>>();
    let offset = ma.len() - mama.len();
    ma.into_iter()
        .skip(offset)
        .zip(mama)
        .map(|(ma1, ma2)| 2.0 * ma1 - ma2)
}

/// triple exponential moving average
/// https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp
pub fn tema(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let ma = ewma(data, window).collect::<Vec<f64>>();
    let ma2 = ewma(&ma, window).collect::<Vec<f64>>();
    let ma3 = ewma(&ma2, window).collect::<Vec<f64>>();
    let ma_offset = ma.len() - ma3.len();
    let ma2_offset = ma2.len() - ma3.len();
    izip!(
        ma.into_iter().skip(ma_offset),
        ma2.into_iter().skip(ma2_offset),
        ma3
    )
    .map(|(ma1, ma2, ma3)| 3.0 * ma1 - 3.0 * ma2 + ma3)
}

/// weighted moving average
pub fn wma(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let denom = (u64::pow(window as u64, 2) + window as u64) as f64 / 2.0;
    let weights: Vec<f64> = (1..=window).map(|i| i as f64 / denom).collect();
    data.windows(window).map(move |w| {
        w.iter()
            .zip(weights.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    })
}

/// welles wilder's moving average
pub fn wilder<I>(data: I, window: usize) -> impl Iterator<Item = f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut data_iter = data.into_iter();
    let initial = data_iter.by_ref().take(window - 1).sum::<f64>() / (window - 1) as f64;
    data_iter.scan(initial, move |state, x| {
        let ma = (*state * (window - 1) as f64 + x) / window as f64;
        *state = ma;
        Some(ma)
    })
}

/// hull's moving average
pub fn hull(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let ma = wma(data, window);
    let ma2 = wma(data, window.div_ceil(2));
    wma(
        &ma2.skip(window / 2)
            .zip(ma)
            .map(|(x, y)| 2.0 * x - y)
            .collect::<Vec<f64>>(),
        (window as f64).sqrt().floor() as usize,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}

/// standard deviation
pub(crate) fn std_dev(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.windows(window).map(move |w| {
        let mean = w.iter().sum::<f64>() / window as f64;
        (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64).sqrt()
    })
}

/// volatility index dynamic average (vidya)
pub fn vidya(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let alpha = 2.0 / (window + 1) as f64;
    let std5 = std_dev(data, 5).collect::<Vec<f64>>();
    let std20 = sma(&std5, 20).collect::<Vec<f64>>();
    let std5_offset = std5.len() - std20.len();
    let data_offset = data.len() - std20.len();
    izip!(
        std20.into_iter(),
        std5.into_iter().skip(std5_offset),
        data.iter().skip(data_offset)
    )
    .scan(0.0, move |state, (s20, s5, d)| {
        *state = alpha * (s5 / s20) * (d - *state) + *state;
        Some(*state)
    })
}

/// chande momentum oscillator
pub(crate) fn _cmo(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    data.windows(2)
        .map(|pair| {
            let (prev, curr) = (pair[0], pair[1]);
            (f64::max(0.0, curr - prev), f64::max(0.0, prev - curr))
        })
        .collect::<Vec<(f64, f64)>>()
        .windows(window)
        .map(|w| {
            let mut gain_sum = 0.0;
            let mut loss_sum = 0.0;
            for (g, l) in w {
                gain_sum += g;
                loss_sum += l;
            }
            (gain_sum - loss_sum) / (gain_sum + loss_sum)
        })
        .collect::<Vec<f64>>()
        .into_iter()
}

/// variable moving average (vma)
pub fn vma(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let alpha = 2.0 / (window + 1) as f64;
    let cmo_win = 9; // maybe make this configurable?
    let vi = _cmo(data, cmo_win);
    izip!(vi, data.iter().skip(cmo_win))
        .scan(0.0, move |state, (vi, d)| {
            *state = alpha * vi.abs() * (d - *state) + *state;
            Some(*state)
        })
        .skip(window.max(cmo_win) - cmo_win)
}

/// Linear Regression Forecast aka Time Series Forecast
/// https://quantstrategy.io/blog/what-is-tsf-understanding-time-series-forecast-indicator/
pub fn lrf<I>(data: I, window: usize) -> impl Iterator<Item = f64>
where
    I: IntoIterator<Item = f64>,
{
    let x_sum = (window * (window + 1)) as f64 / 2.0;
    let x2_sum: f64 = x_sum * (2 * window + 1) as f64 / 3.0;
    let divisor = window as f64 * x2_sum - x_sum.powi(2);
    let indices: Vec<f64> = (1..=window).map(|x| x as f64).collect();

    data.into_iter().collect::<Vec<f64>>().windows(window).map(move |w| {
        let mut y_sum = 0.0;
        let mut xy_sum = 0.0;
        for (count, val) in indices.iter().zip(w.iter()) {
            y_sum += val;
            xy_sum += count * val;
        }
        let m = (window as f64 * xy_sum - x_sum * y_sum) / divisor;
        let b = (y_sum * x2_sum - x_sum * xy_sum) / divisor;
        m * window as f64 + b
    })
    .collect::<Vec<f64>>().into_iter()
}

/// triangular moving average
/// computes sma(N/2) and then sma(N/2) again.
pub fn trima(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let win1 = window.div_ceil(2);
    let win2 = if window & 2 == 0 { win1 + 1 } else { win1 };
    sma(&sma(data, win1).collect::<Vec<f64>>(), win2)
        .collect::<Vec<f64>>()
        .into_iter()
}

/// Zero Lag Moving Average
/// https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
pub fn zlma<I>(data: I, window: usize) -> impl Iterator<Item = f64>
where
    I: IntoIterator<Item = f64>,
{
    let lag = (window - 1) / 2;
    let data_vec = data.into_iter().collect::<Vec<f64>>();
    ewma(
        &data_vec
            .iter()
            .zip(data_vec[lag..].iter())
            .map(|(prev, curr)| 2.0 * curr - prev)
            .collect::<Vec<f64>>(),
        window,
    )
        .collect::<Vec<f64>>().into_iter()
}
