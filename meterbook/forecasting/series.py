"""Point-in-time daily usage series built from reconciled evidence.

A forecasting experiment only trusts usage that was both reconciled and already
*known* at its knowledge cutoff: the head event of each reporting side had been
received before the cutoff.  Days with no such evidence stay "missing" and are
never silently turned into zero, while a day whose known reconciled quantity is
exactly zero is an explicit zero-consumption observation.
"""
import datetime
from decimal import Decimal
from meterbook.values import instant, key, reject, digest, encoded
from meterbook.contracts import registry
from meterbook.events import reconcile, receiver

CHARGEABLE_STATUSES = {'matched', 'within_tolerance', 'resolved'}


def _day(timestamp):
    return instant(timestamp)[:10]


def _parse_day(value):
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        reject('forecast_day', 'Calendar day must use YYYY-MM-DD', value=value)


def day_sequence(first, last):
    start = _parse_day(first)
    end = _parse_day(last)
    if end < start:
        reject('forecast_span', 'Series end precedes its start')
    current = start
    while current <= end:
        yield current.isoformat()
        current += datetime.timedelta(days=1)


def _evidence_known_at(store, contract_id, usage_id, cutoff):
    """Both side head events must have arrived before the cutoff instant."""
    event_ids = []
    for side in ('provider', 'consumer'):
        head = receiver.head(store, contract_id, side, usage_id)
        if head is None:
            return None
        event = receiver.get(store, head['event_id'])
        if event['received_at'] > cutoff:
            return None
        event_ids.append(head['event_id'])
    return event_ids


def build(store, contract_id, run_ids, cutoff_at, meters=None):
    """Construct the frozen daily series and the knowledge provenance per run."""
    registry.get(store, contract_id)
    cutoff = instant(cutoff_at)
    if not isinstance(run_ids, list) or not 1 <= len(run_ids) <= 100:
        reject('forecast_runs', 'Experiment needs one to one hundred reconciliation runs')
    selected = None
    if meters is not None:
        if not isinstance(meters, list) or not meters or len(meters) > 100:
            reject('forecast_meters', 'Meter selection must be a bounded nonempty list')
        selected = sorted(key(meter, 'meter') for meter in meters)
    periods = []
    runs = []
    for run_id in run_ids:
        run = reconcile.get(store, run_id)
        if run['contract_id'] != contract_id:
            reject('forecast_scope', 'Reconciliation belongs to another contract', run_id=run_id)
        if run['period'] in periods:
            reject('forecast_run_overlap', 'Reconciliation runs cover the same period', period=run['period'])
        periods.append(run['period'])
        runs.append(run)
    meter_names = selected
    if meter_names is None:
        meter_names = sorted({row['chosen']['meter'] for run in runs for row in run['rows'] if row['chosen'] is not None})
    buckets = {meter: {} for meter in meter_names}
    evidence = {}
    usage_count = 0
    for run in runs:
        known_ids = []
        for row in run['rows']:
            usage = row['chosen']
            if usage is None or row['status'] not in CHARGEABLE_STATUSES or usage['meter'] not in buckets:
                continue
            event_ids = _evidence_known_at(store, contract_id, row['usage_id'], cutoff)
            if event_ids is None:
                continue
            day = _day(usage['at'])
            if day >= cutoff[:10]:
                continue
            buckets[usage['meter']].setdefault(day, {'quantity': Decimal(0), 'usage_ids': []})
            buckets[usage['meter']][day]['quantity'] += Decimal(usage['quantity'])
            buckets[usage['meter']][day]['usage_ids'].append(row['usage_id'])
            known_ids.append({'usage_id': row['usage_id'], 'events': event_ids, 'day': day, 'status': row['status']})
            usage_count += 1
        evidence[run['run_id']] = {'run_id': run['run_id'], 'period': run['period'], 'digest': run['digest'], 'revision': run['revision'], 'period_revision': run['revision'], 'known_usage': known_ids, 'known_count': len(known_ids)}
    observed_days = [day for meter in meter_names for day in buckets[meter]]
    if not observed_days:
        reject('forecast_history', 'No reconciled usage was known at the cutoff')
    first_day = min(observed_days)
    last_day = (datetime.date.fromisoformat(cutoff[:10]) - datetime.timedelta(days=1)).isoformat()
    if last_day < first_day:
        reject('forecast_history', 'Cutoff leaves no completed history day')
    series = {}
    for meter in meter_names:
        days = []
        zero_days = []
        missing_days = []
        for day in day_sequence(first_day, last_day):
            entry = buckets[meter].get(day)
            if entry is None:
                missing_days.append(day)
                days.append({'day': day, 'status': 'missing', 'quantity': None, 'usage_ids': []})
            else:
                if entry['quantity'] == 0:
                    zero_days.append(day)
                days.append({'day': day, 'status': 'observed', 'quantity': encoded(entry['quantity']), 'usage_ids': sorted(entry['usage_ids'])})
        series[meter] = {'meter': meter, 'first_day': first_day, 'last_day': last_day, 'days': days, 'observed_count': len(days) - len(missing_days), 'zero_days': zero_days, 'missing_days': missing_days}
    provenance = {'contract_id': contract_id, 'cutoff_at': cutoff, 'run_ids': list(run_ids), 'runs': [evidence[run['run_id']] for run in runs], 'known_usage_count': usage_count}
    return {'series': series, 'provenance': provenance, 'digest': digest({'series': series, 'provenance': provenance})}


def observed_window(days, upto_index):
    """Contiguous window ending at upto_index with missing days removed."""
    return [None if point['quantity'] is None else Decimal(point['quantity']) for point in days[:upto_index + 1]]


def origins(days, min_history, max_origins):
    """Evenly spaced rolling-origin indices that keep min_history observations."""
    eligible = [index for index, point in enumerate(days) if point['status'] == 'observed' and index >= min_history - 1]
    if len(eligible) <= max_origins:
        return eligible
    stride = len(eligible) / max_origins
    chosen = []
    used = set()
    for slot in range(max_origins):
        index = eligible[min(int(slot * stride), len(eligible) - 1)]
        if index not in used:
            chosen.append(index)
            used.add(index)
    if eligible[-1] not in used:
        chosen[-1] = eligible[-1]
    return sorted(chosen)
