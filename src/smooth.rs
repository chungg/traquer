use itertools::izip;
use std::iter;

pub enum MaMode {
    SMA,
    EWMA,
    DEMA,
    TEMA,
    WMA,
    Hull,
    LinReg,
    Pascal,
    Triangle,
    Variable,
    Vidya,
    Wilder,
    ZeroLag,
}

pub fn ma(data: &[f64], window: usize, mamode: MaMode) -> Box<dyn Iterator<Item = f64> + '_> {
    match mamode {
        MaMode::SMA => Box::new(sma(data, window)),
        MaMode::EWMA => Box::new(ewma(data, window)),
        MaMode::DEMA => Box::new(dema(data, window)),
        MaMode::TEMA => Box::new(tema(data, window)),
        MaMode::WMA => Box::new(wma(data, window)),
        MaMode::Hull => Box::new(hull(data, window)),
        MaMode::LinReg => Box::new(lrf(data, window)),
        MaMode::Pascal => Box::new(pwma(data, window)),
        MaMode::Triangle => Box::new(trima(data, window)),
        MaMode::Variable => Box::new(vma(data, window)),
        MaMode::Vidya => Box::new(vidya(data, window)),
        MaMode::Wilder => Box::new(wilder(data, window)),
        MaMode::ZeroLag => Box::new(zlma(data, window)),
    }
}

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

/// Pascal's Triangle moving average
/// https://en.wikipedia.org/wiki/Pascal%27s_triangle
pub fn pwma(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let n = window - 1;
    let mut row = vec![1; window];
    let mut denom = 2_usize;
    for i in 1..=(n / 2) {
        let k = i - 1;
        row[i] = (row[k] * (n - k)) / (k + 1);
        row[n - i] = row[i];
        denom += row[i] * 2;
    }
    if window % 2 == 1 {
        denom -= row[n / 2];
    }
    let weights: Vec<f64> = row.into_iter().map(|i| i as f64 / denom as f64).collect();
    data.windows(window).map(move |w| {
        w.iter()
            .zip(weights.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    })
}

/// welles wilder's moving average
pub fn wilder(data: &[f64], window: usize) -> Box<dyn Iterator<Item = f64> + '_> {
    if window == 1 {
        return Box::new(data.iter().copied());
    }
    let initial = data[..window - 1].iter().sum::<f64>() / (window - 1) as f64;
    Box::new(data[window - 1..].iter().scan(initial, move |state, x| {
        let ma = (*state * (window - 1) as f64 + x) / window as f64;
        *state = ma;
        Some(ma)
    }))
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
    iter::once((0.0, 0.0))
        .chain(data.windows(2).scan((0.0, 0.0), |state, pair| {
            let (prev, curr) = (pair[0], pair[1]);
            let gain = state.0 + f64::max(0.0, curr - prev);
            let loss = state.1 + f64::max(0.0, prev - curr);
            *state = (gain, loss);
            Some((gain, loss))
        }))
        .collect::<Vec<(f64, f64)>>()
        .windows(window + 1)
        .map(|w| {
            let gain_sum = w[w.len() - 1].0 - w[0].0;
            let loss_sum = w[w.len() - 1].1 - w[0].1;
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
pub fn lrf(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let x_sum = (window * (window + 1)) as f64 / 2.0;
    let x2_sum: f64 = x_sum * (2 * window + 1) as f64 / 3.0;
    let divisor = window as f64 * x2_sum - x_sum.powi(2);
    let indices: Vec<f64> = (1..=window).map(|x| x as f64).collect();

    data.windows(window).map(move |w| {
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
pub fn zlma(data: &[f64], window: usize) -> impl Iterator<Item = f64> + '_ {
    let lag = (window - 1) / 2;
    ewma(
        &data
            .iter()
            .zip(data[lag..].iter())
            .map(|(prev, curr)| 2.0 * curr - prev)
            .collect::<Vec<f64>>(),
        window,
    )
    .collect::<Vec<f64>>()
    .into_iter()
}
