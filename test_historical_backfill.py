import unittest

import historical_backfill as hb


class HistoricalBackfillTests(unittest.TestCase):
    def test_compute_window_bounds_excludes_open_minute(self):
        now = 10 * hb.TF_MS + 42_000
        start, end = hb.compute_window_bounds(now, lookback_days=0)
        self.assertEqual(end, 9 * hb.TF_MS)
        self.assertEqual(start, 9 * hb.TF_MS)

    def test_build_missing_ranges_single_large_gap(self):
        start = 0
        end = 10 * hb.TF_MS
        existing = {start, end}
        ranges = hb.build_missing_ranges(existing, start, end)
        self.assertEqual(ranges, [(hb.TF_MS, 9 * hb.TF_MS, 9)])

    def test_split_large_ranges_chunks_to_cap(self):
        start = 0
        end = 10 * hb.TF_MS
        ranges = [(start, end, 11)]
        chunks = hb.split_large_ranges(ranges, max_gap_minutes=4)
        self.assertEqual(
            chunks,
            [
                (0, 3 * hb.TF_MS, 4),
                (4 * hb.TF_MS, 7 * hb.TF_MS, 4),
                (8 * hb.TF_MS, 10 * hb.TF_MS, 3),
            ],
        )


if __name__ == '__main__':
    unittest.main()
