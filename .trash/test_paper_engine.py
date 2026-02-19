import unittest

from db import connect, init_db
import paper_engine as pe


class PaperEngineTests(unittest.TestCase):
    def setUp(self):
        self.con = connect(':memory:')
        init_db(self.con)

    def test_open_trade_gets_live_unrealized_pnl(self):
        now = pe.now_ms()
        self.con.execute(
            """
            INSERT INTO paper_trades(
                confirmed_signal_id, opened_ts_ms, opened_ts_iso, symbol, side, entry_price, qty, status, reason
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (1, now, pe.iso(now), 'BTC-USDT', 'BUY', 100.0, 1.0, 'OPEN', 'ENTRY: test'),
        )
        self.con.execute(
            "INSERT INTO ticks(ts_ms, ts_iso, symbol, price, source, raw_json) VALUES(?,?,?,?,?,?)",
            (now + 1, pe.iso(now + 1), 'BTC-USDT', 100.5, 'test', '{}'),
        )

        closed = pe.close_paper_trades(self.con)
        self.assertEqual(closed, 0)

        row = self.con.execute("SELECT status, pnl_pct FROM paper_trades WHERE confirmed_signal_id=1").fetchone()
        self.assertEqual(row['status'], 'OPEN')
        expected = pe.calc_pnl_pct_with_friction(100.0, 100.5, 'BUY')
        self.assertAlmostEqual(float(row['pnl_pct']), expected, places=6)

    def test_short_trade_uses_friction_model(self):
        now = pe.now_ms()
        self.con.execute(
            """
            INSERT INTO paper_trades(
                confirmed_signal_id, opened_ts_ms, opened_ts_iso, symbol, side, entry_price, qty, status, reason
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (2, now, pe.iso(now), 'BTC-USDT', 'SELL', 100.0, 1.0, 'OPEN', 'ENTRY: test short'),
        )
        self.con.execute(
            "INSERT INTO ticks(ts_ms, ts_iso, symbol, price, source, raw_json) VALUES(?,?,?,?,?,?)",
            (now + 1, pe.iso(now + 1), 'BTC-USDT', 99.5, 'test', '{}'),
        )

        closed = pe.close_paper_trades(self.con)
        self.assertEqual(closed, 0)

        row = self.con.execute("SELECT status, pnl_pct FROM paper_trades WHERE confirmed_signal_id=2").fetchone()
        self.assertEqual(row['status'], 'OPEN')
        expected = pe.calc_pnl_pct_with_friction(100.0, 99.5, 'SELL')
        self.assertAlmostEqual(float(row['pnl_pct']), expected, places=6)


if __name__ == '__main__':
    unittest.main()
