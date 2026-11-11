import datetime
from tests.support import Case
from meterbook.service import Meterbook
from meterbook.contracts import registry, rates
from meterbook.events import receiver, reconcile


def day(offset, start='2026-09-01'):
    return (datetime.date.fromisoformat(start) + datetime.timedelta(days=offset)).isoformat()


class ForecastCase(Case):
    def setUp(self):
        # Clock sits inside the seeded history window so that evidence is already
        # "known" at the default experiment cutoff; explicit cutoffs perform the
        # point-in-time filtering independently of this value.
        self.app = Meterbook(clock=lambda: '2026-09-20T12:00:00.000000Z')
        self.store = self.app.store
        from meterbook.events import disputes
        disputes.register(self.store, 'analyst', 'analyst')
        disputes.register(self.store, 'approver', 'approver')

    def contract_with(self, meters, tariffs=None):
        registry.create(self.store, 'contract-f', 'provider-f', 'consumer-f', 'CNY', {
            'valid_from': '2026-01-01T00:00:00Z', 'valid_until': '2027-01-01T00:00:00Z',
            'meters': meters, 'purposes': ['research'], 'regions': ['CN'],
            'tolerance': {meter: '0' for meter in meters}})
        tariffs = tariffs or {meter: {'tiers': [{'up_to': None, 'unit_price': '1'}]} for meter in meters}
        for meter, tariff in tariffs.items():
            rates.add(self.store, 'rate-' + meter, 'contract-f', meter,
                      '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z', tariff)

    def pair_on(self, seq, meter, day_value, quantity, contract='contract-f', identity=None):
        identity = identity or f'{meter}-{seq}'
        for side in ('provider', 'consumer'):
            event = {'event_id': f'{side}-{contract}-{seq}', 'contract_id': contract, 'side': side,
                     'sequence': seq, 'usage_id': identity, 'revision': 1, 'action': 'report',
                     'usage': {'meter': meter, 'at': f'{day_value}T12:00:00Z', 'quantity': str(quantity),
                               'purpose': 'research', 'region': 'CN', 'dimensions': {}}}
            receiver.receive(self.store, event)
            result = receiver.drain(self.store, contract, side)
            self.assertIsNone(result['blocked'])

    def correct_on(self, meter, day_value, quantity, identity, contract='contract-f'):
        for side in ('provider', 'consumer'):
            status = receiver.stream_status(self.store, contract, side)
            seq = status['maximum_received'] + 1
            event = {'event_id': f'{side}-{contract}-corr-{seq}', 'contract_id': contract, 'side': side,
                     'sequence': seq, 'usage_id': identity, 'revision': 2, 'action': 'correct',
                     'usage': {'meter': meter, 'at': f'{day_value}T12:00:00Z', 'quantity': str(quantity),
                               'purpose': 'research', 'region': 'CN', 'dimensions': {}},
                     'reason': 'late correction'}
            receiver.receive(self.store, event)
            result = receiver.drain(self.store, contract, side)
            self.assertIsNone(result['blocked'])

    def reconcile(self, run_id='run-f', period='2026-09'):
        return reconcile.run(self.store, run_id, 'contract-f', period)

    def advance_clock(self, instant_value):
        self.store.clock = lambda: instant_value

    def definition(self, **changes):
        result = {'run_ids': ['run-f'], 'cutoff_at': '2026-10-01T00:00:00Z',
                  'horizon_days': 7, 'forecast_period': '2026-10',
                  'min_history': 8, 'max_origins': 6, 'interval_percent': 80}
        result.update(changes)
        return result

    def load(self, experiment_id):
        import json
        return json.loads(self.app.forecasts.get(experiment_id)['result'])
