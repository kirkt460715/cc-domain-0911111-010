"""Savable usage-forecast experiments with frozen rates and full lineage.

Two kinds of follow-up experiments are deliberately separate:

* ``revise`` rebuilds the daily series from later reconciled evidence (for
  example after a late usage correction), reruns every model and backtest, and
  keeps the frozen rate book untouched so the two versions can be compared.
* ``reprice`` never retrains.  It reuses a stored experiment's exact forecasts
  and converts them again with a newly frozen rate schedule only.

Every stored window records the knowledge version (per-run revisions, digests
and known usage identities) that its training window could actually see.
"""
import json
from decimal import Decimal
from meterbook.values import fields, key, number, instant, canonical, digest, reject, encoded, prose, MeterError
from meterbook.contracts import registry
from meterbook.events import reconcile
from meterbook.forecasting import models, series as series_module, backtest, money

DDL = '''
CREATE TABLE IF NOT EXISTS forecast_experiments(experiment_id TEXT PRIMARY KEY,contract_id TEXT NOT NULL,parent_id TEXT,kind TEXT NOT NULL,state TEXT NOT NULL,definition TEXT NOT NULL,digest TEXT NOT NULL,rate_book TEXT NOT NULL,rate_fingerprint TEXT NOT NULL,result TEXT NOT NULL,result_digest TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS forecast_history(experiment_id TEXT NOT NULL,revision INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(experiment_id,revision));
'''

MIN_HISTORY_DEFAULT = 14
MAX_ORIGINS_DEFAULT = 12
HORIZON_LIMIT = 92
WINDOW_LIMIT = 3660
INTERVAL_LEVELS = (50, 80, 90, 95)


def normalize_definition(value):
    fields(value, {'run_ids', 'cutoff_at', 'horizon_days', 'forecast_period'}, {'meters', 'min_history', 'max_origins', 'interval_percent'})
    run_ids = value['run_ids']
    if not isinstance(run_ids, list) or not 1 <= len(run_ids) <= 100 or any(not isinstance(item, str) for item in run_ids):
        reject('forecast_runs', 'Experiment needs a bounded list of run identifiers')
    if len(set(run_ids)) != len(run_ids):
        reject('forecast_runs', 'Reconciliation runs must not repeat')
    horizon = number(value['horizon_days'], 'horizon_days', 1, HORIZON_LIMIT)
    min_history = number(value.get('min_history', MIN_HISTORY_DEFAULT), 'min_history', 2, WINDOW_LIMIT)
    max_origins = number(value.get('max_origins', MAX_ORIGINS_DEFAULT), 'max_origins', 1, 1000)
    percent = value.get('interval_percent', 80)
    if percent not in INTERVAL_LEVELS:
        reject('forecast_interval', 'Interval level must be one of the supported percents', allowed=list(INTERVAL_LEVELS))
    month = value['forecast_period']
    from meterbook.values import month as month_validator
    month_validator(month)
    meters = None
    if 'meters' in value:
        if not isinstance(value['meters'], list) or not value['meters'] or len(value['meters']) > 100:
            reject('forecast_meters', 'Meter selection must be a bounded nonempty list')
        meters = sorted(key(item, 'meter') for item in value['meters'])
    return {'run_ids': list(run_ids), 'cutoff_at': instant(value['cutoff_at']), 'horizon_days': horizon, 'forecast_period': month, 'meters': meters, 'min_history': min_history, 'max_origins': max_origins, 'interval_percent': percent}


class Forecasts:
    def __init__(self, store):
        self.store = store
        if store.depth:
            reject('forecast_transaction', 'Initialize forecast experiments outside transactions')
        store.db.executescript(DDL)

    def _forecast_days(self, cutoff, horizon):
        import datetime
        first = datetime.date.fromisoformat(cutoff[:10])
        return [(first + datetime.timedelta(days=offset)).isoformat() for offset in range(horizon)]

    def _final_fit(self, window, horizon, chosen, backtests):
        """Fit the selected model on the full window, falling back on gaps."""
        ordered = [chosen] + [model for model in models.MODEL_ORDER if model != chosen]
        last_error = None
        for model in ordered:
            if not backtests[model]['available'] or not models.available(window, model):
                continue
            try:
                return model, models.fit(window, horizon, model)
            except MeterError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        reject('forecast_window', 'No model can fit the full window')

    def _run_models(self, definition, built, forecast_days=None):
        if forecast_days is None:
            forecast_days = self._forecast_days(definition['cutoff_at'], definition['horizon_days'])
        meter_outputs = {}
        degradations = []
        for meter, meter_series in sorted(built['series'].items()):
            window = series_module.observed_window(meter_series['days'], len(meter_series['days']) - 1)
            backtests = backtest.backtest_meter(meter_series, built['provenance'], definition['horizon_days'], definition['min_history'], definition['max_origins'], definition['interval_percent'])
            selection = backtest.choose(backtests)
            total = sum(Decimal(point['quantity']) for point in meter_series['days'] if point['status'] == 'observed')
            if not total:
                degradations.append({'meter': meter, 'code': 'all_zero_history', 'message': 'Every observed day is zero; forecasts are pinned to zero'})
                daily = [{'day': day, 'point': '0', 'lower': '0', 'upper': '0', 'model': 'degraded_zero'} for day in forecast_days]
                meter_outputs[meter] = {'degraded': True, 'degradation': 'all_zero_history', 'selection': None, 'backtest': backtests, 'daily': daily, 'observed_count': meter_series['observed_count'], 'zero_days': meter_series['zero_days'], 'missing_days': meter_series['missing_days']}
                continue
            if selection is None:
                degradations.append({'meter': meter, 'code': 'insufficient_history', 'message': 'No model has enough observations; latest value is used without intervals'})
                fitted = models.fit(window, definition['horizon_days'], models.LATEST)
                daily = [{'day': forecast_days[index], 'point': item['forecast'], 'lower': item['forecast'], 'upper': item['forecast'], 'model': models.LATEST} for index, item in enumerate(fitted['forecast'])]
                meter_outputs[meter] = {'degraded': True, 'degradation': 'insufficient_history', 'selection': None, 'backtest': backtests, 'daily': daily, 'observed_count': meter_series['observed_count'], 'zero_days': meter_series['zero_days'], 'missing_days': meter_series['missing_days']}
                continue
            chosen_model, fitted = self._final_fit(window, definition['horizon_days'], selection['selected'], backtests)
            chosen_report = backtests[chosen_model]
            daily = []
            for index, item in enumerate(fitted['forecast']):
                step = item['step']
                point = Decimal(item['forecast'])
                band_entry = next((row for row in chosen_report['residual_intervals'] if row['step'] == step), None)
                if band_entry is None:
                    lower = upper = point
                else:
                    lower = max(Decimal(0), point + Decimal(band_entry['lower_delta']))
                    upper = max(Decimal(0), point + Decimal(band_entry['upper_delta']))
                daily.append({'day': forecast_days[index], 'point': encoded(point), 'lower': encoded(lower), 'upper': encoded(upper), 'model': chosen_model})
            if chosen_model != selection['selected']:
                degradations.append({'meter': meter, 'code': 'model_fallback', 'message': 'Selected model could not fit the gappy full window', 'selected': selection['selected'], 'used': chosen_model})
            meter_outputs[meter] = {'degraded': False, 'degradation': None, 'selection': selection, 'backtest': backtests, 'daily': daily, 'observed_count': meter_series['observed_count'], 'zero_days': meter_series['zero_days'], 'missing_days': meter_series['missing_days']}
        return meter_outputs, forecast_days, degradations

    def _assemble(self, definition, built, frozen, meter_outputs, forecast_days, kind, parent_id):
        book = frozen['book']
        cost_reports = {}
        for meter in sorted(built['series']):
            cost_reports[meter] = money.meter_cost_paths(book, meter, meter_outputs[meter]['daily'])
        combined = money.combine_paths(list(cost_reports.values()))
        observed_total_cents, observed_per_meter = money.observed_cost(book, built['series'])
        result = {'kind': kind, 'parent_id': parent_id, 'definition': definition, 'knowledge': built['provenance'], 'series_digest': built['digest'], 'rate_fingerprint': frozen['fingerprint'], 'observed_series': {meter: [point for point in meter_series['days'] if point['status'] == 'observed'] for meter, meter_series in built['series'].items()}, 'forecast_days': forecast_days, 'meters': meter_outputs, 'costs': {'per_meter': cost_reports, 'combined': combined, 'observed_history_cents': observed_total_cents, 'observed_history_per_meter_cents': observed_per_meter}, 'degradations': self._degradation_summary(meter_outputs)}
        return result

    def _degradation_summary(self, meter_outputs):
        summary = []
        for meter, output in sorted(meter_outputs.items()):
            if output['degraded']:
                summary.append({'meter': meter, 'code': output['degradation'], 'observed_count': output['observed_count'], 'zero_days': len(output['zero_days']), 'missing_days': len(output['missing_days'])})
        return summary

    def create(self, experiment_id, contract_id, definition):
        key(experiment_id)
        normalized = normalize_definition(definition)
        registry.get(self.store, contract_id)
        with self.store.atomic():
            if self.store.one('SELECT experiment_id FROM forecast_experiments WHERE experiment_id=?', (experiment_id,)):
                reject('forecast_exists', 'Forecast experiment identifier exists')
            built = series_module.build(self.store, contract_id, normalized['run_ids'], normalized['cutoff_at'], normalized['meters'])
            forecast_days = self._forecast_days(normalized['cutoff_at'], normalized['horizon_days'])
            frozen = money.freeze_rates(self.store, contract_id, sorted(built['series']), forecast_days)
            meter_outputs, forecast_days, degradations = self._run_models(normalized, built)
            result = self._assemble(normalized, built, frozen, meter_outputs, forecast_days, 'forecast', None)
            self._persist(experiment_id, contract_id, None, 'forecast', normalized, frozen, result)
            self.store.log('forecast.created', {'experiment_id': experiment_id, 'contract_id': contract_id, 'degradations': degradations})
            return self.get(experiment_id)

    def revise(self, new_experiment_id, parent_id, reason, run_ids=None, cutoff_at=None):
        key(new_experiment_id)
        reason = prose(reason, 'reason')
        with self.store.atomic():
            parent = self.get(parent_id)
            if parent['state'] != 'frozen':
                reject('forecast_state', 'Only frozen experiments can spawn revisions')
            if self.store.one('SELECT experiment_id FROM forecast_experiments WHERE experiment_id=?', (new_experiment_id,)):
                reject('forecast_exists', 'Forecast experiment identifier exists')
            definition = dict(parent['definition'])
            contract_id = parent['contract_id']
            # The knowledge cutoff may move forward to absorb late evidence, but
            # the forecast anchor (and thus forecast days) stays aligned with the
            # parent so the two versions remain directly comparable.
            forecast_days = list(json.loads(parent['result'])['forecast_days'])
            new_cutoff = instant(cutoff_at) if cutoff_at is not None else parent['definition']['cutoff_at']
            if new_cutoff < parent['definition']['cutoff_at']:
                reject('forecast_cutoff', 'A revision cannot move the knowledge cutoff backwards')
            definition['cutoff_at'] = new_cutoff
            if run_ids is not None:
                if not isinstance(run_ids, list) or not run_ids or len(run_ids) > 100:
                    reject('forecast_runs', 'Revision needs a bounded list of run identifiers')
                old_periods = sorted(row['period'] for row in json.loads(parent['result'])['knowledge']['runs'])
                new_periods = sorted(reconcile.get(self.store, run_id)['period'] for run_id in run_ids)
                if new_periods != old_periods:
                    reject('forecast_run_scope', 'Revision runs must cover the same periods as the parent experiment')
                definition['run_ids'] = list(run_ids)
            built = series_module.build(self.store, contract_id, definition['run_ids'], definition['cutoff_at'], definition['meters'])
            stored_book = json.loads(parent['rate_book'])
            if digest(stored_book) != parent['rate_fingerprint']:
                reject('forecast_rate_corrupt', 'Stored frozen rate book checksum differs')
            frozen = {'book': stored_book, 'fingerprint': parent['rate_fingerprint']}
            missing = [day for meter in stored_book['rates'].values() for day in forecast_days if not any(row['start_at'] <= day + 'T12:00:00Z' < row['end_at'] for row in meter)]
            if missing:
                reject('forecast_rate_uncovered', 'Frozen rate book does not cover the revised horizon', days=missing[:10])
            meter_outputs, forecast_days, degradations = self._run_models(definition, built, forecast_days=forecast_days)
            result = self._assemble(definition, built, frozen, meter_outputs, forecast_days, 'revision', parent_id)
            result['revision_reason'] = reason
            self._persist(new_experiment_id, contract_id, parent_id, 'revision', definition, frozen, result)
            self.store.log('forecast.revised', {'experiment_id': new_experiment_id, 'parent_id': parent_id, 'reason': reason, 'degradations': degradations})
            return self.get(new_experiment_id)

    def reprice(self, new_experiment_id, parent_id, reason):
        key(new_experiment_id)
        reason = prose(reason, 'reason')
        with self.store.atomic():
            parent = self.get(parent_id)
            if parent['state'] != 'frozen':
                reject('forecast_state', 'Only frozen experiments can be repriced')
            if self.store.one('SELECT experiment_id FROM forecast_experiments WHERE experiment_id=?', (new_experiment_id,)):
                reject('forecast_exists', 'Forecast experiment identifier exists')
            definition = dict(parent['definition'])
            contract_id = parent['contract_id']
            old_result = json.loads(parent['result'])
            if digest(old_result) != parent['result_digest']:
                reject('forecast_result_corrupt', 'Stored forecast result checksum differs')
            built_provenance = old_result['knowledge']
            forecast_days = old_result['forecast_days']
            frozen = money.freeze_rates(self.store, contract_id, sorted(old_result['meters']), forecast_days)
            meter_outputs = old_result['meters']
            cost_reports = {}
            for meter in sorted(meter_outputs):
                cost_reports[meter] = money.meter_cost_paths(frozen['book'], meter, meter_outputs[meter]['daily'])
            combined = money.combine_paths(list(cost_reports.values()))
            observed_series = {meter: {'days': days} for meter, days in old_result['observed_series'].items()}
            observed_total_cents, observed_per_meter = money.observed_cost(frozen['book'], observed_series)
            result = {'kind': 'reprice', 'parent_id': parent_id, 'definition': definition, 'knowledge': built_provenance, 'series_digest': old_result['series_digest'], 'rate_fingerprint': frozen['fingerprint'], 'observed_series': old_result['observed_series'], 'forecast_days': forecast_days, 'meters': old_result['meters'], 'costs': {'per_meter': cost_reports, 'combined': combined, 'observed_history_cents': observed_total_cents, 'observed_history_per_meter_cents': observed_per_meter}, 'degradations': old_result['degradations'], 'reprice_reason': reason, 'retrained': False}
            self._persist(new_experiment_id, contract_id, parent_id, 'reprice', definition, frozen, result)
            self.store.log('forecast.repriced', {'experiment_id': new_experiment_id, 'parent_id': parent_id, 'reason': reason, 'old_rate_fingerprint': parent['rate_fingerprint'], 'new_rate_fingerprint': frozen['fingerprint']})
            return self.get(new_experiment_id)

    def budget_impact(self, experiment_id, budget_id):
        with self.store.atomic():
            experiment = self.get(experiment_id)
            budget_row = self.store.one('SELECT * FROM budgets WHERE budget_id=?', (budget_id,))
            if budget_row is None:
                reject('budget_missing', 'Budget does not exist')
            budget_definition = json.loads(budget_row['definition'])
            if digest(budget_definition) != budget_row['digest']:
                reject('budget_corrupt', 'Budget definition checksum mismatch')
            result = json.loads(experiment['result'])
            definition = result['definition']
            if (budget_row['contract_id'], budget_row['period']) != (experiment['contract_id'], definition['forecast_period']):
                reject('forecast_budget_scope', 'Budget and forecast target different contract or period')
            closed_cents = money.closed_for_period(self.store, experiment['contract_id'], budget_row['period'])
            observed_unclosed = max(0, result['costs']['observed_history_cents'] - closed_cents)
            impact = money.budget_window(self.store, experiment['contract_id'], budget_row['period'], budget_definition['maximum_cents'], closed_cents, observed_unclosed, result['costs']['combined'])
            payload = {'experiment_id': experiment_id, 'budget_id': budget_id, 'budget_revision': budget_row['revision'], **impact, 'combined_total_cents': result['costs']['combined']['total_cents']}
            payload['digest'] = digest({key: value for key, value in payload.items() if key != 'digest'})
            self.store.log('forecast.budget_impact', {'experiment_id': experiment_id, 'budget_id': budget_id, 'breach': impact['breach_within_period']})
            return payload

    def compare(self, left_id, right_id):
        left = self.get(left_id)
        right = self.get(right_id)
        if left['contract_id'] != right['contract_id']:
            reject('forecast_scope', 'Experiments belong to different contracts')
        left_result = json.loads(left['result'])
        right_result = json.loads(right['result'])
        changes = []
        meters = sorted(set(left_result['meters']) | set(right_result['meters']))
        for meter in meters:
            a = {row['day']: row for row in left_result['meters'].get(meter, {}).get('daily', [])}
            b = {row['day']: row for row in right_result['meters'].get(meter, {}).get('daily', [])}
            for day in sorted(set(a) | set(b)):
                before = a.get(day)
                after = b.get(day)
                if before != after:
                    changes.append({'meter': meter, 'day': day, 'before': before, 'after': after})
        return {'left': left_id, 'right': right_id, 'left_kind': left_result['kind'], 'right_kind': right_result['kind'], 'series_changed': left_result['series_digest'] != right_result['series_digest'], 'rates_changed': left_result['rate_fingerprint'] != right_result['rate_fingerprint'], 'left_total_cents': left_result['costs']['combined']['total_cents'], 'right_total_cents': right_result['costs']['combined']['total_cents'], 'delta_total_cents': {path: right_result['costs']['combined']['total_cents'][path] - left_result['costs']['combined']['total_cents'][path] for path in ('point', 'lower', 'upper')}, 'daily_changes': changes}

    def _persist(self, experiment_id, contract_id, parent_id, kind, definition, frozen, result):
        result_digest = digest(result)
        book = frozen['book']
        self.store.execute("INSERT INTO forecast_experiments(experiment_id,contract_id,parent_id,kind,state,definition,digest,rate_book,rate_fingerprint,result,result_digest,created_at) VALUES(?,?,?,?,'frozen',?,?,?,?,?,?,?)", (experiment_id, contract_id, parent_id, kind, canonical(definition), digest(definition), canonical(book), frozen['fingerprint'], canonical(result), result_digest, self.store.clock()))
        self.store.execute('INSERT INTO forecast_history VALUES(?,?,?)', (experiment_id, 1, canonical({'experiment_id': experiment_id, 'definition': definition, 'result_digest': result_digest, 'rate_fingerprint': frozen['fingerprint']})))

    def get(self, experiment_id):
        row = self.store.one('SELECT * FROM forecast_experiments WHERE experiment_id=?', (experiment_id,))
        if row is None:
            reject('forecast_missing', 'Forecast experiment does not exist')
        row['definition'] = json.loads(row['definition'])
        if digest(row['definition']) != row['digest']:
            reject('forecast_corrupt', 'Experiment definition checksum differs')
        result = json.loads(row['result'])
        if digest(result) != row['result_digest']:
            reject('forecast_result_corrupt', 'Stored forecast result checksum differs')
        book = json.loads(row['rate_book'])
        if digest(book) != row['rate_fingerprint']:
            reject('forecast_rate_corrupt', 'Frozen rate book checksum differs')
        return row

    def listing(self, contract_id=None):
        if contract_id:
            registry.get(self.store, contract_id)
            rows = self.store.all('SELECT experiment_id FROM forecast_experiments WHERE contract_id=? ORDER BY created_at,experiment_id', (contract_id,))
        else:
            rows = self.store.all('SELECT experiment_id FROM forecast_experiments ORDER BY created_at,experiment_id')
        return [self.get(row['experiment_id']) for row in rows]

    def lineage(self, experiment_id):
        """Knowledge version used by every training/backtest window."""
        experiment = self.get(experiment_id)
        result = json.loads(experiment['result'])
        windows = []
        for meter, output in sorted(result['meters'].items()):
            for model, report in sorted(output['backtest'].items()):
                for window in report['windows']:
                    windows.append({'meter': meter, 'model': model, 'origin_day': window['origin_day'], 'training_observations': window['training_observations'], 'parameters': window['parameters'], 'knowledge_digest': window['knowledge']['digest'], 'knowledge': window['knowledge']})
        return {'experiment_id': experiment_id, 'cutoff_at': result['definition']['cutoff_at'], 'run_knowledge': result['knowledge']['runs'], 'windows': windows, 'window_count': len(windows)}


def check(store):
    """Verify every stored experiment's frozen digests and history rows."""
    exists = store.one("SELECT name FROM sqlite_master WHERE type='table' AND name='forecast_experiments'")
    if not exists:
        return {'configured': False, 'problems': []}
    problems = []
    for row in store.all('SELECT experiment_id,definition,digest,rate_book,rate_fingerprint,result,result_digest FROM forecast_experiments ORDER BY experiment_id'):
        experiment_id = row['experiment_id']
        try:
            definition = json.loads(row['definition'])
            if digest(definition) != row['digest']:
                problems.append({'code': 'forecast_definition_corrupt', 'experiment_id': experiment_id})
            book = json.loads(row['rate_book'])
            if digest(book) != row['rate_fingerprint']:
                problems.append({'code': 'forecast_rate_corrupt', 'experiment_id': experiment_id})
            result = json.loads(row['result'])
            if digest(result) != row['result_digest']:
                problems.append({'code': 'forecast_result_corrupt', 'experiment_id': experiment_id})
            history = store.all('SELECT payload FROM forecast_history WHERE experiment_id=? ORDER BY revision', (experiment_id,))
            expected_history = {'experiment_id': experiment_id, 'definition': definition, 'result_digest': row['result_digest'], 'rate_fingerprint': row['rate_fingerprint']}
            if not history or json.loads(history[-1]['payload']) != expected_history:
                problems.append({'code': 'forecast_history_corrupt', 'experiment_id': experiment_id})
        except MeterError as exc:
            problems.append(exc.payload())
    return {'configured': True, 'problems': problems, 'ok': not problems}
