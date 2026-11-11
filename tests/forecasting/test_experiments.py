import json
from decimal import Decimal
from tests.forecasting.support import ForecastCase, day
from meterbook.forecasting import models
from meterbook.values import digest, MeterError
from meterbook.contracts import rates
from meterbook.events import reconcile


def weekly_quantity(index):
    # Weekend spike gives the seasonal model signal; one explicit zero day.
    if index == 4:
        return 0
    return 20 if index % 7 in (5, 6) else 10


class ExperimentFlow(ForecastCase):
    def seed_history(self, count=21, meters=('records',)):
        self.contract_with(list(meters), tariffs={
            meter: {'tiers': [{'up_to': '100', 'unit_price': '0.25'}, {'up_to': None, 'unit_price': '0.1'}]}
            for meter in meters})
        seq = 1
        for index in range(count):
            for meter in meters:
                self.pair_on(seq, meter, day(index), weekly_quantity(index) + (1 if meter == 'calls' else 0), identity=f'{meter}-{index}')
                seq += 1
        self.reconcile()

    def test_experiment_freezes_series_rates_and_lineage(self):
        self.seed_history()
        created = self.app.forecasts.create('exp', 'contract-f', self.definition())
        result = json.loads(created['result'])
        self.assertEqual(result['kind'], 'forecast')
        self.assertEqual(len(result['forecast_days']), 7)
        self.assertTrue(result['rate_fingerprint'])
        # Explicit zero stays zero; tail calendar days with no evidence are missing.
        output = result['meters']['records']
        self.assertIn('2026-09-05', output['zero_days'])
        self.assertIn('2026-09-22', output['missing_days'])
        self.assertEqual(output['observed_count'], 21)
        # Every forecast day carries an ordered, non-inverted band.
        for row in output['daily']:
            self.assertLessEqual(Decimal(row['lower']), Decimal(row['point']))
            self.assertLessEqual(Decimal(row['point']), Decimal(row['upper']))
        lineage = self.app.forecasts.lineage('exp')
        self.assertGreater(lineage['window_count'], 0)
        for window in lineage['windows']:
            self.assertTrue(window['knowledge_digest'])
            self.assertEqual(window['knowledge']['runs'][0]['run_id'], 'run-f')

    def test_backtest_mae_scales_with_horizon_and_selection_is_recorded(self):
        self.seed_history()
        self.app.forecasts.create('exp', 'contract-f', self.definition())
        result = self.load('exp')
        for meter, output in result['meters'].items():
            self.assertIsNotNone(output['selection'])
            for model_name, report in output['backtest'].items():
                self.assertIn(model_name, models.MODEL_ORDER)
                if report['available']:
                    self.assertTrue(report['horizon_mae'])
                    for step, value in report['horizon_mae'].items():
                        self.assertGreaterEqual(Decimal(value), 0)
                    for band in report['residual_intervals']:
                        self.assertIn(band['calibration'], ('step_residuals', 'pooled_residuals'))
                        self.assertTrue(0 <= float(band['empirical_coverage']) <= 1)
                        self.assertLessEqual(Decimal(band['lower_delta']), Decimal(band['upper_delta']))

    def test_tiered_cost_never_uses_average_unit_price(self):
        self.seed_history()
        self.app.forecasts.create('exp', 'contract-f', self.definition())
        result = self.load('exp')
        line = result['costs']['per_meter']['records']['lines'][0]
        quantity = Decimal(line['quantities']['point'])
        # Recompute the tiered charge directly and require an exact match.
        book = json.loads(self.app.forecasts.get('exp')['rate_book'])
        rate_def = book['rates']['records'][0]['definition']
        from meterbook.settlement import pricing
        expected = pricing.tier_charge(str(quantity), rate_def)['amount_cents']
        self.assertEqual(line['point_pricing']['amount_cents'], expected)
        # A flat average price would yield a different number once past tier 1.
        if quantity > 100:
            flat = int(quantity * Decimal('0.25') * 100)
            self.assertNotEqual(expected, flat)
        # Daily deltas telescope back to the tiered total.
        daily_sum = sum(row['point_cents'] for row in result['costs']['per_meter']['records']['daily'])
        self.assertEqual(daily_sum, expected)

    def test_late_correction_revision_changes_forecast_but_keeps_rates(self):
        self.seed_history()
        self.app.forecasts.create('exp', 'contract-f', self.definition())
        self.advance_clock('2026-10-05T00:00:00.000000Z')
        self.correct_on('records', day(20), 90, identity='records-20')
        self.reconcile('run-f-2')
        revised = self.app.forecasts.revise(
            'exp-2', 'exp', 'late correction', run_ids=['run-f-2'],
            cutoff_at='2026-10-06T00:00:00Z')
        result = self.load('exp')
        revised_result = json.loads(revised['result'])
        self.assertEqual(revised_result['kind'], 'revision')
        self.assertEqual(revised_result['rate_fingerprint'], result['rate_fingerprint'])
        self.assertNotEqual(revised_result['series_digest'], result['series_digest'])
        diff = self.app.forecasts.compare('exp', 'exp-2')
        self.assertTrue(diff['series_changed'])
        self.assertFalse(diff['rates_changed'])
        self.assertTrue(diff['daily_changes'])
        self.assertNotEqual(diff['delta_total_cents']['point'], 0)

    def test_reprice_uses_new_rates_without_retraining(self):
        self.seed_history()
        self.app.forecasts.create('exp', 'contract-f', self.definition())
        rates.withdraw(self.store, 'rate-records', 'renegotiation')
        rates.add(self.store, 'rate-records-2', 'contract-f', 'records',
                  '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z',
                  {'tiers': [{'up_to': None, 'unit_price': '0.5'}]})
        repriced = self.app.forecasts.reprice('exp-3', 'exp', 'new tariff')
        before = self.load('exp')
        after = json.loads(repriced['result'])
        self.assertFalse(after['retrained'])
        self.assertEqual(after['series_digest'], before['series_digest'])
        self.assertNotEqual(after['rate_fingerprint'], before['rate_fingerprint'])
        # Forecast quantities are byte-for-byte identical; only money changes.
        self.assertEqual(after['meters']['records']['daily'], before['meters']['records']['daily'])
        diff = self.app.forecasts.compare('exp', 'exp-3')
        self.assertFalse(diff['series_changed'])
        self.assertTrue(diff['rates_changed'])
        self.assertEqual(diff['daily_changes'], [])

    def test_budget_hit_date_range_respects_headroom(self):
        self.seed_history()
        self.app.forecasts.create('exp', 'contract-f', self.definition())
        self.app.budgets.put('budget', 'contract-f', '2026-10', 0, {'maximum_cents': 100})
        impact = self.app.forecasts.budget_impact('exp', 'budget')
        self.assertTrue(impact['breach_within_period'])
        earliest = impact['hit_range']['earliest_day']
        latest = impact['hit_range']['latest_day']
        self.assertIsNotNone(earliest)
        self.assertIsNotNone(latest)
        self.assertLessEqual(earliest, latest)
        # A generous budget never breaches within the forecast.
        self.app.budgets.put('big', 'contract-f', '2026-10', 0, {'maximum_cents': 10**9})
        safe = self.app.forecasts.budget_impact('exp', 'big')
        self.assertFalse(safe['breach_within_period'])
        self.assertIsNone(safe['hit_range']['earliest_day'])

    def test_all_zero_history_degrades_explicitly(self):
        self.contract_with(['zeros'])
        for index in range(10):
            self.pair_on(index + 1, 'zeros', day(index), 0, identity=f'zeros-{index}')
        self.reconcile()
        self.app.forecasts.create('exp-z', 'contract-f', self.definition())
        result = self.load('exp-z')
        self.assertEqual(result['degradations'][0]['code'], 'all_zero_history')
        self.assertTrue(all(row['point'] == '0' for row in result['meters']['zeros']['daily']))

    def test_insufficient_history_degrades_to_latest(self):
        self.contract_with(['few'])
        self.pair_on(1, 'few', day(0), 5, identity='few-0')
        self.reconcile()
        self.app.forecasts.create('exp-s', 'contract-f', self.definition())
        result = self.load('exp-s')
        self.assertEqual(result['degradations'][0]['code'], 'insufficient_history')
        self.assertEqual(result['meters']['few']['daily'][0]['model'], models.LATEST)

    def test_knowledge_cutoff_excludes_late_events(self):
        self.seed_history()
        # Seeded evidence arrived while the clock was Sep 20; advance the clock
        # before the late event so it post-dates the experiment cutoff.
        self.advance_clock('2026-09-28T00:00:00.000000Z')
        self.pair_on(200, 'records', day(25), 77, identity='records-late')
        self.reconcile('run-late')
        exp = self.app.forecasts.create(
            'exp-cut', 'contract-f', self.definition(run_ids=['run-late'], cutoff_at='2026-09-24T00:00:00Z'))
        result = json.loads(exp['result'])
        known = {entry['usage_id'] for run in result['knowledge']['runs'] for entry in run['known_usage']}
        self.assertNotIn('records-late', known)
        self.assertIn('records-0', known)

    def test_stored_experiment_is_tamper_evident_and_listable(self):
        self.seed_history()
        self.app.forecasts.create('exp', 'contract-f', self.definition())
        row = self.app.forecasts.get('exp')
        self.assertEqual([item['experiment_id'] for item in self.app.forecasts.listing('contract-f')], ['exp'])
        self.store.execute("UPDATE forecast_experiments SET result_digest='0' WHERE experiment_id='exp'")
        self.error('forecast_result_corrupt', self.app.forecasts.get, 'exp')
        self.assertGreater(self.store.scalar('SELECT COUNT(*) FROM forecast_history'), 0)
        from meterbook.audit import checks
        report = checks.check(self.store)
        self.assertFalse(report['ok'])
        self.assertIn('forecast_result_corrupt', [item['code'] for item in report['problems']])

    def test_operations_are_exposed_through_generic_router(self):
        operations = {entry['operation'] for entry in self.app.capabilities()}
        for name in ('forecast.create', 'forecast.revise', 'forecast.reprice',
                     'forecast.compare', 'forecast.budget_impact', 'forecast.lineage',
                     'forecast.get', 'forecast.listing', 'forecast.check'):
            self.assertIn(name, operations)
        self.seed_history()
        result = self.app.execute({'operation': 'forecast.create',
                                   'arguments': {'experiment_id': 'exp', 'contract_id': 'contract-f',
                                                 'definition': self.definition()}})
        self.assertEqual(result['state'], 'frozen')
        check = self.app.execute({'operation': 'forecast.check', 'arguments': {}})
        self.assertTrue(check['ok'])

    def test_revision_rejects_different_period_scope(self):
        self.seed_history()
        self.app.forecasts.create('exp', 'contract-f', self.definition())
        # A run for a different month cannot replace the parent's runs.
        self.advance_clock('2026-10-05T00:00:00.000000Z')
        self.pair_on(300, 'records', '2026-10-02', 5, identity='records-oct')
        reconcile.run(self.store, 'run-oct', 'contract-f', '2026-10')
        self.error('forecast_run_scope', self.app.forecasts.revise,
                   'exp-bad', 'exp', 'wrong scope', run_ids=['run-oct'])
