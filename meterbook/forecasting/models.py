"""Deterministic daily-usage forecasting primitives.

All arithmetic uses :class:`~decimal.Decimal` so experiments are reproducible
without floating point noise.  The three model families deliberately share one
interface: a fit takes a contiguous observed window (missing days removed) and
returns the in-sample residuals plus an h-step-ahead point forecast.

Every tunable parameter is derived from the training window alone.  The
exponential smoother searches a fixed grid and never sees future observations.
"""
from decimal import Decimal, ROUND_HALF_EVEN
from meterbook.values import encoded, reject

LATEST = 'latest'
SEASONAL = 'seasonal_naive'
EXPONENTIAL = 'exponential_smoothing'
MODEL_ORDER = (LATEST, SEASONAL, EXPONENTIAL)

WEEK = 7
MIN_SEASONAL_HISTORY = WEEK + 1
MIN_SMOOTHING_HISTORY = 2
SMOOTHING_GRID = ('0.05', '0.1', '0.2', '0.3', '0.5', '0.7', '0.9')
QUANT = Decimal('0.000001')


def quantize(value):
    """Keep every fitted level on the canonical six-decimal usage scale."""
    return value.quantize(QUANT, rounding=ROUND_HALF_EVEN)


def available(window, model):
    """Return whether a model has enough non-missing observations."""
    observed = [point for point in window if point is not None]
    if not observed:
        return False
    if model == LATEST:
        return True
    if model == SEASONAL:
        return len(observed) >= MIN_SEASONAL_HISTORY
    if model == EXPONENTIAL:
        return len(observed) >= MIN_SMOOTHING_HISTORY
    reject('forecast_model', 'Unknown forecasting model', model=model)


def _step(day_index):
    """Forecast horizon step (one-based) for a zero-based day index."""
    return day_index + 1


def latest_fit(window, horizon):
    observed = [(index, value) for index, value in enumerate(window) if value is not None]
    if not observed:
        reject('forecast_window', 'Latest model needs an observed value')
    anchor_index, anchor = observed[-1]
    forecasts = []
    for day in range(len(window), len(window) + horizon):
        forecasts.append({'step': _step(day - anchor_index), 'forecast': encoded(max(Decimal(0), anchor))})
    residuals = [{'index': index, 'residual': encoded(value - anchor)} for index, value in observed[:-1]]
    return {'model': LATEST, 'parameters': {'anchor': encoded(anchor)}, 'forecast': forecasts, 'residuals': residuals}


def seasonal_fit(window, horizon):
    observed = [(index, value) for index, value in enumerate(window) if value is not None]
    if len(observed) < MIN_SEASONAL_HISTORY:
        reject('forecast_window', 'Seasonal naive needs at least eight observations')
    values = {index: value for index, value in observed}
    forecasts = []
    for day in range(len(window), len(window) + horizon):
        anchor = day - WEEK
        if anchor not in values:
            reject('forecast_seasonal', 'Same-week anchor is missing', day=day)
        forecasts.append({'step': _step(day - len(window) + 1), 'forecast': encoded(max(Decimal(0), values[anchor]))})
    residuals = [{'index': index, 'residual': encoded(value - values[index - WEEK])} for index, value in observed if index - WEEK in values]
    return {'model': SEASONAL, 'parameters': {'week': WEEK}, 'forecast': forecasts, 'residuals': residuals}


def _smoothing_sse(observed, alpha):
    level = observed[0]
    sse = Decimal(0)
    for value in observed[1:]:
        sse += (value - level) ** 2
        level = quantize(alpha * value + (Decimal(1) - alpha) * level)
    return sse, level


def exponential_fit(window, horizon):
    observed = [value for value in window if value is not None]
    if len(observed) < MIN_SMOOTHING_HISTORY:
        reject('forecast_window', 'Exponential smoothing needs two observations')
    best = None
    for candidate in SMOOTHING_GRID:
        alpha = Decimal(candidate)
        sse, level = _smoothing_sse(observed, alpha)
        if best is None or sse < best['sse']:
            best = {'alpha': alpha, 'sse': sse, 'level': level}
    alpha = best['alpha']
    level = best['level']
    point = max(Decimal(0), level)
    forecasts = [{'step': step, 'forecast': encoded(quantize(point))} for step in range(1, horizon + 1)]
    residuals = []
    running = observed[0]
    for index, value in enumerate(observed[1:], 1):
        residuals.append({'index': index, 'residual': encoded(quantize(value - running))})
        running = quantize(alpha * value + (Decimal(1) - alpha) * running)
    return {'model': EXPONENTIAL, 'parameters': {'alpha': encoded(alpha), 'grid': list(SMOOTHING_GRID), 'objective': 'in_sample_sse'}, 'forecast': forecasts, 'residuals': residuals}


def fit(window, horizon, model):
    if model == LATEST:
        return latest_fit(window, horizon)
    if model == SEASONAL:
        return seasonal_fit(window, horizon)
    if model == EXPONENTIAL:
        return exponential_fit(window, horizon)
    reject('forecast_model', 'Unknown forecasting model', model=model)


def nearest_rank(values, probability):
    """Nearest-rank quantile (CEILING(p*n)) over Decimals, deterministic."""
    ordered = sorted(values)
    rank = int((probability * len(ordered)).to_integral_value(rounding='ROUND_CEILING'))
    rank = max(1, min(rank, len(ordered)))
    return ordered[rank - 1]


def interval(residuals, point, percent):
    """Residual quantile interval, clamped so quantities never go negative."""
    probability = Decimal(percent) / Decimal(100)
    if not residuals:
        return {'level_percent': percent, 'sample_size': 0, 'lower': encoded(point), 'upper': encoded(point), 'degraded': True, 'reason': 'no_residuals'}
    lower_delta = nearest_rank(residuals, Decimal(1) - probability)
    upper_delta = nearest_rank(residuals, probability)
    return {'level_percent': percent, 'sample_size': len(residuals), 'lower': encoded(max(Decimal(0), point + lower_delta)), 'upper': encoded(max(Decimal(0), point + upper_delta)), 'degraded': False, 'reason': 'residual_quantiles'}


def mean_absolute_error(pairs):
    """MAE over (forecast, actual) Decimal pairs; None when empty."""
    if not pairs:
        return None
    total = sum(abs(forecast - actual) for forecast, actual in pairs)
    return quantize(total / Decimal(len(pairs)))
