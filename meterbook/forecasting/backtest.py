"""Rolling-origin backtesting and residual-based interval calibration.

Each origin refits every candidate model on data available *up to that origin
only*; parameters therefore come exclusively from the corresponding training
window.  Errors are grouped by forecast step so the reported MAE scales with the
forecast horizon, and historical residuals drive empirical quantile intervals
whose realized coverage is measured honestly: the interval used at an origin is
calibrated from the earlier origins only.  Steps with too few residuals fall
back to the model's pooled residual deltas so a thin long-horizon sample can
never produce an inverted band.
"""
from decimal import Decimal
from meterbook.values import encoded, digest, MeterError
from meterbook.forecasting import models, series

MIN_STEP_SAMPLE = 5


def _knowledge(provenance, origin_day):
    """Knowledge version effectively visible when an origin window closed."""
    cutoff = origin_day + 'T23:59:59.999999Z'
    runs = []
    for run in provenance['runs']:
        known = [entry for entry in run['known_usage'] if entry['day'] <= origin_day]
        runs.append({'run_id': run['run_id'], 'period': run['period'], 'digest': run['digest'], 'revision': run['revision'], 'known_count_at_origin': len(known), 'known_usage_ids': [entry['usage_id'] for entry in known]})
    return {'cutoff_at': cutoff, 'runs': runs, 'digest': digest({'origin_day': origin_day, 'runs': runs})}


def _band(deltas, percent):
    probability = Decimal(percent) / Decimal(100)
    lower = models.nearest_rank(deltas, Decimal(1) - probability)
    upper = models.nearest_rank(deltas, probability)
    return lower, upper


def backtest_meter(meter_series, provenance, horizon, min_history, max_origins, interval_percent):
    origin_indices = series.origins(meter_series['days'], min_history, max_origins)
    days = meter_series['days']
    report = {}
    for model in models.MODEL_ORDER:
        windows = []
        step_records = {}
        step_errors = {}
        pooled_deltas = []
        for origin in origin_indices:
            window = series.observed_window(days, origin)
            if not models.available(window, model):
                continue
            try:
                fitted = models.fit(window, horizon, model)
            except MeterError:
                # A same-week anchor can still be missing inside a gappy window.
                continue
            pairs = []
            for prediction in fitted['forecast']:
                step = prediction['step']
                target_index = origin + step
                if target_index >= len(days) or days[target_index]['status'] != 'observed':
                    continue
                forecast = Decimal(prediction['forecast'])
                actual = Decimal(days[target_index]['quantity'])
                delta = forecast - actual
                pairs.append({'step': step, 'day': days[target_index]['day'], 'forecast': encoded(forecast), 'actual': encoded(actual), 'abs_error': encoded(abs(forecast - actual)), 'residual_delta': encoded(delta)})
                step_errors.setdefault(step, []).append((forecast, actual))
                step_records.setdefault(step, []).append({'origin_day': days[origin]['day'], 'forecast': forecast, 'actual': actual, 'delta': delta})
                pooled_deltas.append(delta)
            if not pairs:
                continue
            windows.append({'origin_day': days[origin]['day'], 'training_observations': sum(point is not None for point in window), 'training_first_day': days[0]['day'], 'parameters': fitted['parameters'], 'pairs': pairs, 'window_mae': encoded(models.mean_absolute_error([(Decimal(pair['forecast']), Decimal(pair['actual'])) for pair in pairs])), 'knowledge': _knowledge(provenance, days[origin]['day'])})

        intervals = []
        for step in sorted(step_records):
            records = step_records[step]
            if not records:
                continue
            step_deltas = [row['delta'] for row in records]
            inside = 0
            band_sources = []
            for index, record in enumerate(records):
                prior = [row['delta'] for row in records[:index]]
                if len(prior) >= MIN_STEP_SAMPLE:
                    lower, upper = _band(prior, interval_percent)
                    source = 'step'
                elif pooled_deltas:
                    lower, upper = _band(pooled_deltas, interval_percent)
                    source = 'pooled'
                else:
                    lower = upper = Decimal(0)
                    source = 'none'
                band_sources.append(source)
                if record['forecast'] + lower <= record['actual'] <= record['forecast'] + upper:
                    inside += 1
            full_lower, full_upper = _band(step_deltas if len(step_deltas) >= MIN_STEP_SAMPLE else pooled_deltas, interval_percent)
            intervals.append({'step': step, 'sample_size': len(records), 'lower_delta': encoded(full_lower), 'upper_delta': encoded(full_upper), 'calibration': 'step_residuals' if len(step_deltas) >= MIN_STEP_SAMPLE else 'pooled_residuals', 'empirical_coverage': encoded(models.quantize(Decimal(inside) / Decimal(len(records)))), 'band_sources': band_sources, 'degraded': len(step_deltas) < MIN_STEP_SAMPLE, 'reason': 'residual_quantiles'})

        horizon_mae = {}
        all_pairs = []
        for step in sorted(step_errors):
            if step_errors[step]:
                horizon_mae[str(step)] = encoded(models.mean_absolute_error(step_errors[step]))
                all_pairs.extend(step_errors[step])
        report[model] = {'model': model, 'window_count': len(windows), 'pair_count': len(all_pairs), 'mae': encoded(models.mean_absolute_error(all_pairs)) if all_pairs else None, 'horizon_mae': horizon_mae, 'residual_intervals': intervals, 'windows': windows, 'available': bool(windows)}
    return report


def choose(backtest):
    """Select the candidate with the lowest overall backtest MAE."""
    ranked = [entry for entry in backtest.values() if entry['available'] and entry['mae'] is not None]
    if not ranked:
        return None
    ranking = {model: priority for priority, model in enumerate(models.MODEL_ORDER)}
    best = min(ranked, key=lambda entry: (Decimal(entry['mae']), ranking[entry['model']]))
    others = [{'model': entry['model'], 'mae': entry['mae'], 'pair_count': entry['pair_count']} for entry in ranked if entry['model'] != best['model']]
    return {'selected': best['model'], 'criterion': 'lowest_overall_rolling_origin_mae', 'tie_break_order': list(models.MODEL_ORDER), 'selected_mae': best['mae'], 'candidates': others}
