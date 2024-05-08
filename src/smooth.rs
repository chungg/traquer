use itertools::izip;

/// exponentially weighted moving average
pub fn ewma(data: &[f64], window: u8) -> Vec<f64> {
    let initial = data[..window as usize].iter().sum::<f64>() / window as f64;
    let alpha = 2.0 / (window + 1) as f64;
    let mut result = data[window as usize..]
        .iter()
        .scan(initial, |state, &x| {
            *state = x * alpha + *state * (1.0 - alpha);
            Some(*state)
        })
        .collect::<Vec<f64>>();
    result.insert(0, initial);
    result
}

/// simple moving average
pub fn sma(data: &[f64], window: u8) -> Vec<f64> {
    data.windows(window.into())
        .map(|w| w.iter().sum::<f64>() / window as f64)
        .collect::<Vec<f64>>()
}

/// double exponential moving average
pub fn dema(data: &[f64], window: u8) -> Vec<f64> {
    let ma = ewma(data, window);
    let mama = ewma(&ma, window);
    ma[ma.len() - mama.len()..]
        .iter()
        .zip(mama.iter())
        .map(|(ma1, ma2)| 2.0 * ma1 - ma2)
        .collect::<Vec<f64>>()
}

/// weighted moving average
pub fn wma(data: &[f64], window: u8) -> Vec<f64> {
    let denom = (u8::pow(window, 2) + window) as f64 / 2.0;
    data.windows(window.into())
        .map(|w| {
            w.iter()
                .enumerate()
                .map(|(i, e)| (e * (i + 1) as f64 / denom))
                .sum()
        })
        .collect::<Vec<f64>>()
}

/// welles wilder's moving average
pub fn wilder(data: &Vec<f64>, window: u8) -> Vec<f64> {
    data[window as usize..]
        .iter()
        .scan(
            data[..window as usize].iter().sum::<f64>() / window as f64,
            |state, x| {
                let ma = (*state * (window - 1) as f64 + x) / window as f64;
                *state = ma;
                Some(ma)
            },
        )
        .collect::<Vec<f64>>()
}

/// hull's moving average
pub fn hull(data: &Vec<f64>, window: u8) -> Vec<f64> {
    let ma = wma(&data, window);
    let ma2 = wma(&data, u8::div_ceil(window, 2));
    wma(
        &ma2[(ma2.len() - ma.len())..]
            .iter()
            .zip(ma.iter())
            .map(|(x, y)| 2.0 * x - y)
            .collect::<Vec<f64>>(),
        (window as f64).sqrt().floor() as u8,
    )
}

/// standard deviation
fn std_dev(data: &Vec<f64>, window: u8) -> Vec<f64> {
    data.windows(window.into())
        .map(|w| {
            let mean = w.iter().sum::<f64>() / window as f64;
            (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64).sqrt()
        })
        .collect::<Vec<f64>>()
}

/// volatility index dynamic average (vidya)
pub fn vidya(data: &Vec<f64>, window: u8) -> Vec<f64> {
    let alpha = 2.0 / (window + 1) as f64;
    let std5 = std_dev(data, 5);
    let std20 = sma(&std5, 20);
    izip!(
        &std20,
        &std5[std5.len() - std20.len()..],
        &data[data.len() - std20.len()..]
    )
    .scan(0.0, |state, (s20, s5, d)| {
        *state = alpha * (s5 / s20) * (d - *state) + *state;
        Some(*state)
    })
    .collect::<Vec<f64>>()
}

/// chande momentum oscillator
pub(crate) fn _cmo(data: &Vec<f64>, window: u8) -> Vec<f64> {
    let (gain, loss): (Vec<f64>, Vec<f64>) = data[..data.len() - 1]
        .iter()
        .zip(data[1..].iter())
        .map(|(x, y)| (f64::max(0.0, x - y), f64::max(0.0, y - x)))
        .unzip();
    gain.windows(window.into())
        .zip(loss.windows(window.into()))
        .map(|(g, l)| {
            let gain_sum = g.iter().sum::<f64>();
            let loss_sum = l.iter().sum::<f64>();
            (gain_sum - loss_sum) / (gain_sum + loss_sum)
        })
        .collect::<Vec<f64>>()
}

/// variable moving average (vma)
pub fn vma(data: &Vec<f64>, window: u8) -> Vec<f64> {
    let alpha = 2.0 / (window + 1) as f64;
    let vi = _cmo(data, 9); // maybe make this configurable?
    izip!(&vi, &data[data.len() - vi.len()..])
        .scan(0.0, |state, (vi, d)| {
            *state = alpha * vi.abs() * (d - *state) + *state;
            Some(*state)
        })
        .collect::<Vec<f64>>()
}
