"""Convert frozen quantity forecasts into money without averaging unit prices.

Rates are snapshotted when an experiment is created and every forecast day must
resolve to exactly one active rate interval.  Multi-tier schedules are applied
to cumulative meter quantities at the rate-interval granularity (the same
billing scope as settlement pricing); daily figures are derived as deltas of
the cumulative tiered charge, so they telescope back to the tiered total instead
of pretending a flat average price exists.
"""
import datetime
from decimal import Decimal
from meterbook.values import encoded, reject, digest
from meterbook.contracts import rates, registry
from meterbook.settlement import pricing, invoices


def freeze_rates(store, contract_id, meters, forecast_days):
    registry.get(store, contract_id)
    book = {meter: [] for meter in meters}
    for rate in rates.listing(store, contract_id):
        if rate['state'] != 'active' or rate['meter'] not in book:
            continue
        book[rate['meter']].append({'rate_id': rate['rate_id'], 'meter': rate['meter'], 'start_at': rate['start_at'], 'end_at': rate['end_at'], 'definition': rate['definition'], 'digest': rate['digest']})
    for meter in sorted(book):
        book[meter].sort(key=lambda row: row['start_at'])
    for meter, days in book.items():
        for day in forecast_days:
            stamp = day + 'T12:00:00Z'
            matches = [row for row in days if row['start_at'] <= stamp < row['end_at']]
            if len(matches) != 1:
                reject('forecast_rate_uncovered', 'Forecast day must resolve to exactly one frozen rate', meter=meter, day=day, candidates=len(matches))
    stored = {'contract_id': contract_id, 'rates': book}
    return {'book': stored, 'fingerprint': digest(stored)}


def _interval_for(book, meter, stamp):
    for row in book['rates'][meter]:
        if row['start_at'] <= stamp < row['end_at']:
            return row
    reject('forecast_rate_uncovered', 'Frozen rate book does not cover timestamp', meter=meter, at=stamp)


def meter_cost_paths(book, meter, daily_rows):
    """Tiered cumulative charges per rate interval for each quantity path."""
    paths = ('point', 'lower', 'upper')
    intervals = {}
    for row in daily_rows:
        stamp = row['day'] + 'T12:00:00Z'
        rate = _interval_for(book, meter, stamp)
        bucket = intervals.setdefault(rate['rate_id'], {'rate': rate, 'rows': []})
        bucket['rows'].append(row)
    result_days = {}
    totals = {path: 0 for path in paths}
    lines = []
    for rate_id in sorted(intervals):
        bucket = intervals[rate_id]
        cumulative = {path: Decimal(0) for path in paths}
        previous_cents = {path: 0 for path in paths}
        definition = bucket['rate']['definition']
        for row in bucket['rows']:
            day_cost = {}
            for path in paths:
                cumulative[path] += Decimal(row[path])
                current_cents = pricing.tier_charge(encoded(cumulative[path]), definition)['amount_cents']
                day_cost[path + '_cents'] = current_cents - previous_cents[path]
                previous_cents[path] = current_cents
                totals[path] += day_cost[path + '_cents']
            result_days[row['day']] = {'rate_id': rate_id, **day_cost}
        quantities = {path: encoded(cumulative[path]) for path in paths}
        lines.append({'rate_id': rate_id, 'rate_digest': bucket['rate']['digest'], 'meter': meter, 'interval': [bucket['rate']['start_at'], bucket['rate']['end_at']], 'quantities': quantities, 'point_pricing': pricing.tier_charge(quantities['point'], definition)})
    ordered = [{'day': day, **result_days[day]} for day in sorted(result_days)]
    return {'meter': meter, 'daily': ordered, 'total_cents': totals, 'lines': lines}


def combine_paths(meter_reports):
    days = {}
    totals = {'point': 0, 'lower': 0, 'upper': 0}
    for report in meter_reports:
        for row in report['daily']:
            entry = days.setdefault(row['day'], {'day': row['day'], 'point_cents': 0, 'lower_cents': 0, 'upper_cents': 0})
            for path in totals:
                delta = row[path + '_cents']
                entry[path + '_cents'] += delta
                totals[path] += delta
    return {'daily': [days[day] for day in sorted(days)], 'total_cents': totals}


def observed_cost(book, meter_series):
    """Tiered cost of the known reconciled history inside the frozen book."""
    totals = {}
    for meter, series_row in meter_series.items():
        observed = [{'day': point['day'], 'point': point['quantity'], 'lower': point['quantity'], 'upper': point['quantity']} for point in series_row['days'] if point['status'] == 'observed']
        if not observed:
            totals[meter] = 0
            continue
        totals[meter] = meter_cost_paths(book, meter, observed)['total_cents']['point']
    return sum(totals.values()), totals


def _month_end(period):
    year, month_number = map(int, period.split('-'))
    if month_number == 12:
        next_month = datetime.date(year + 1, 1, 1)
    else:
        next_month = datetime.date(year + 1, month_number + 1, 1)
    return (next_month - datetime.timedelta(days=1)).isoformat()


def budget_window(store, contract_id, period, maximum_cents, closed_cents, observed_unclosed_cents, combined):
    """Earliest/latest budget breach dates using upper/lower cost paths."""
    headroom = maximum_cents - closed_cents - observed_unclosed_cents
    final_day = _month_end(period)
    hits = {}
    for path, field in (('earliest', 'upper_cents'), ('point', 'point_cents'), ('latest', 'lower_cents')):
        cumulative = 0
        hit_day = None
        for row in combined['daily']:
            if row['day'] > final_day:
                break
            cumulative += row[field]
            if closed_cents + observed_unclosed_cents + cumulative > maximum_cents:
                hit_day = row['day']
                break
        hits[path] = hit_day
    return {'contract_id': contract_id, 'period': period, 'maximum_cents': maximum_cents, 'closed_cents': closed_cents, 'observed_unclosed_cents': observed_unclosed_cents, 'headroom_cents': headroom, 'point_hit_day': hits['point'], 'hit_range': {'earliest_day': hits['earliest'], 'latest_day': hits['latest']}, 'breach_within_period': hits['earliest'] is not None}


def closed_for_period(store, contract_id, period):
    chain = invoices.chain(store, contract_id, period)
    return chain['cumulative_cents']
