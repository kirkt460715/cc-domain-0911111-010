import datetime
import unittest
from decimal import Decimal
from meterbook.forecasting import models


def day(offset, start='2026-09-01'):
    return (datetime.date.fromisoformat(start) + datetime.timedelta(days=offset)).isoformat()


class PrimitiveModels(unittest.TestCase):
    def test_latest_uses_only_last_observation(self):
        window = [Decimal(1), None, Decimal(3), Decimal(5)]
        fit = models.latest_fit(window, 2)
        self.assertEqual([row['forecast'] for row in fit['forecast']], ['5', '5'])
        self.assertEqual(fit['parameters'], {'anchor': '5'})

    def test_seasonal_naive_picks_same_weekday(self):
        window = [Decimal(i + 1) for i in range(10)]
        fit = models.seasonal_fit(window, 3)
        # forecast for index 10 -> value at index 3; 11 -> 4; 12 -> 5
        self.assertEqual([row['forecast'] for row in fit['forecast']], ['4', '5', '6'])
        self.assertEqual(len(fit['residuals']), 3)

    def test_exponential_smoothing_alpha_is_window_selected(self):
        window = [Decimal(10), Decimal(10), Decimal(10), Decimal(10), Decimal(10)]
        flat = models.exponential_fit(window, 1)
        self.assertEqual(Decimal(flat['forecast'][0]['forecast']), Decimal('10'))
        rising = models.exponential_fit([Decimal(i) for i in range(1, 12)], 1)
        self.assertIn(rising['parameters']['alpha'], models.SMOOTHING_GRID)
        self.assertEqual(Decimal(rising['parameters']['alpha']), Decimal('0.9'))

    def test_smoothing_keeps_canonical_six_decimal_scale(self):
        window = [Decimal('10.123456'), Decimal('12.654321'), Decimal('9.000001')]
        fit = models.exponential_fit(window, 2)
        for row in fit['forecast']:
            self.assertGreaterEqual(Decimal(row['forecast']).as_tuple().exponent, -6)

    def test_quantile_band_never_inverts(self):
        residuals = [Decimal(-2), Decimal(-1), Decimal(0), Decimal(1), Decimal(2), Decimal(3)]
        band = models.interval(residuals, Decimal(10), 80)
        self.assertLessEqual(Decimal(band['lower']), Decimal(band['upper']))
        self.assertEqual(band['sample_size'], 6)

    def test_availability_and_mae(self):
        self.assertTrue(models.available([Decimal(1)], models.LATEST))
        self.assertFalse(models.available([Decimal(1)] * 7, models.SEASONAL))
        self.assertTrue(models.available([Decimal(1)] * 8, models.SEASONAL))
        self.assertFalse(models.available([Decimal(1)], models.EXPONENTIAL))
        mae = models.mean_absolute_error([(Decimal(2), Decimal(5)), (Decimal(0), Decimal(0))])
        self.assertEqual(mae, Decimal('1.500000'))
        self.assertIsNone(models.mean_absolute_error([]))
