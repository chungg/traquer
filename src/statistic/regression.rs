//! Regression functions
//!
//! Provides functions for estimating the relationships between a dependent variable
//! and one or more independent variables. Potentially useful for forecasting and
//! determing causal relationships.[[1]](https://en.wikipedia.org/wiki/Regression_analysis)
use std::iter;

/// Mean Squared Error
///
/// Measures average squared difference between estimated and actual values.
///
/// ```math
/// MSE = \frac{1}{T}\sum_{t=1}^{T} (X(t) - \hat{X}(t))^2
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Mean_squared_error)
///
/// # Examples
///
/// ```
/// use traquer::statistic::regression;
///
/// regression::mse(&vec![1.0,2.0,3.0,4.0,5.0], &vec![1.0,2.0,3.0,4.0,5.0]).collect::<Vec<f64>>();
/// ```
pub fn mse<'a>(data: &'a [f64], estimate: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    data.iter()
        .enumerate()
        .zip(estimate)
        .scan(0.0, |state, ((cnt, observe), est)| {
            *state += (observe - est).powi(2).max(0.0);
            Some(*state / (cnt + 1) as f64)
        })
}

/// Root Mean Squared Error
///
/// Square root of MSE, but normalises it to same units as input.
///
/// ```math
/// RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (Y_i – \hat{Y}_i)^2}
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Root_mean_square_deviation)
///
/// # Examples
///
/// ```
/// use traquer::statistic::regression;
///
/// regression::rmse(&vec![1.0,2.0,3.0,4.0,5.0], &vec![1.0,2.0,3.0,4.0,5.0]).collect::<Vec<f64>>();
/// ```
pub fn rmse<'a>(data: &'a [f64], estimate: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    data.iter()
        .enumerate()
        .zip(estimate)
        .scan(0.0, |state, ((cnt, observe), est)| {
            *state += (observe - est).powi(2).max(0.0);
            Some((*state / (cnt + 1) as f64).sqrt())
        })
}

/// Mean Absolute Error
///
/// Measures the average magnitude of the errors in a set of predictions. Less sensitive to
/// outliers compared to MSE and RMSE.
///
/// ```math
/// MAE = \frac{1}{n} \sum_{i=1}^{n} |Y_i – \hat{Y}_i|
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Mean_absolute_error)
///
/// # Examples
///
/// ```
/// use traquer::statistic::regression;
///
/// regression::mae(&vec![1.0,2.0,3.0,4.0,5.0], &vec![1.0,2.0,3.0,4.0,5.0]).collect::<Vec<f64>>();
/// ```
pub fn mae<'a>(data: &'a [f64], estimate: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    data.iter()
        .enumerate()
        .zip(estimate)
        .scan(0.0, |state, ((cnt, observe), est)| {
            *state += (observe - est).abs().max(0.0);
            Some(*state / (cnt + 1) as f64)
        })
}

/// Mean Absolute Percentage Error
///
/// Measures the error as a percentage of the actual value.
///
/// ```math
/// MAPE = \frac{100%}{n} \sum_{i=1}^{n} \left|\frac{Y_i – \hat{Y}_i}{Y_i}\right|
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
///
/// # Examples
///
/// ```
/// use traquer::statistic::regression;
///
/// regression::mape(&vec![1.0,2.0,3.0,4.0,5.0], &vec![1.0,2.0,3.0,4.0,5.0]).collect::<Vec<f64>>();
/// ```
pub fn mape<'a>(data: &'a [f64], estimate: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    data.iter()
        .enumerate()
        .zip(estimate)
        .scan(0.0, |state, ((cnt, observe), est)| {
            *state += ((observe - est) / observe).abs().max(0.0);
            Some(100.0 * *state / (cnt + 1) as f64)
        })
}

/// Symmetric Mean Absolute Percentage Error
///
/// Similar to MAPE but attempts to limit the overweighting of negative errors in MAPE. Computed so
/// range is bound to between 0 and 100.
///
/// ```math
/// SMAPE = \frac{100}{n} \sum_{t=1}^n \frac{|F_t-A_t|}{|A_t|+|F_t|}
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
///
/// # Examples
///
/// ```
/// use traquer::statistic::regression;
///
/// regression::smape(&vec![1.0,2.0,3.0,4.0,5.0], &vec![1.0,2.0,3.0,4.0,5.0]).collect::<Vec<f64>>();
/// ```
pub fn smape<'a>(data: &'a [f64], estimate: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    data.iter()
        .enumerate()
        .zip(estimate)
        .scan(0.0, |state, ((cnt, observe), est)| {
            *state += ((observe - est).abs() / ((observe.abs() + est.abs()) / 2.0)).max(0.0);
            Some(100.0 * *state / (cnt + 1) as f64)
        })
}

/// Mean Directional Accuracy
///
/// Measure of prediction accuracy of a forecasting method in statistics. It compares the forecast
/// direction (upward or downward) to the actual realized direction.
///
/// ```math
/// MDA = \frac{1}{N}\sum_t \mathbf{1}_{\sgn(A_t - A_{t-1}) = \sgn(F_t - A_{t-1})}
/// ```
///
/// ## Sources
///
/// [[1]](https://en.wikipedia.org/wiki/Mean_directional_accuracy)
///
/// # Examples
///
/// ```
/// use traquer::statistic::regression;
///
/// regression::mda(&vec![1.0,2.0,3.0,4.0,5.0], &vec![1.0,2.0,3.0,4.0,5.0]).collect::<Vec<f64>>();
/// ```
pub fn mda<'a>(data: &'a [f64], estimate: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    iter::once(f64::NAN).chain(data[1..].iter().enumerate().zip(&estimate[1..]).scan(
        (0.0, data[0]),
        |state, ((cnt, observe), est)| {
            let dir = ((observe - state.1).signum() == (est - state.1).signum()) as u8 as f64;
            *state = (state.0 + dir, *observe);
            Some(state.0 / (cnt + 1) as f64)
        },
    ))
}
