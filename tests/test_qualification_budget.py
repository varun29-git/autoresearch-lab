"""CPU regression tests for the campaign allocation ledger."""

import json
from pathlib import Path
import tempfile
import unittest

from production.kimi_h100_5b import campaign
from production.kimi_h100_5b.budget import BudgetExhausted, Ledger, LedgerError


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class LedgerTests(unittest.TestCase):
    def test_open_consume_close_and_resume_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            clock = FakeClock()
            ledger = Ledger(path, clock=clock)
            ledger.open_allocation(note="vm allocated")
            clock.advance(3600)
            self.assertAlmostEqual(ledger.consumed_hours(), 1.0)
            self.assertAlmostEqual(ledger.remaining_hours(), 49.0)
            ledger.log_activity("trial_start", {"trial": "H"})
            clock.advance(3600)
            ledger.close_allocation()
            # A new process re-reads the file: elapsed time is never reset.
            clock.advance(600)
            resumed = Ledger(path, clock=clock)
            self.assertAlmostEqual(resumed.consumed_hours(), 2.0)
            self.assertFalse(resumed.allocation_open)
            resumed.open_allocation()
            clock.advance(1800)
            self.assertAlmostEqual(resumed.consumed_hours(), 2.5)

    def test_require_fails_closed_with_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            clock = FakeClock()
            ledger = Ledger(path, clock=clock)
            ledger.open_allocation()
            clock.advance(41 * 3600)
            status = ledger.require(8.5, label="five proxies", reserve_hours=0.5)
            self.assertTrue(status["fits"])
            with self.assertRaisesRegex(BudgetExhausted, "five proxies"):
                ledger.require(8.6, label="five proxies", reserve_hours=0.5)
            with self.assertRaises(ValueError):
                ledger.require(float("nan"), label="bad")
            with self.assertRaises(ValueError):
                ledger.require(-1.0, label="bad")

    def test_malformed_and_unbalanced_ledgers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            path.write_text('{"record": "allocation_open", "time_unix": 1}\nnot json\n')
            with self.assertRaises(LedgerError):
                Ledger(path)
            path.write_text('{"record": "mystery", "time_unix": 1}\n')
            with self.assertRaises(LedgerError):
                Ledger(path)
            path.write_text('{"record": "allocation_open", "time_unix": 1}\n'
                            '{"record": "allocation_open", "time_unix": 2}\n')
            with self.assertRaises(LedgerError):
                Ledger(path)

    def test_double_open_close_without_open_and_cap_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            ledger = Ledger(Path(directory) / "budget.jsonl", clock=clock)
            with self.assertRaisesRegex(LedgerError, "no allocation"):
                ledger.close_allocation()
            ledger.open_allocation()
            with self.assertRaisesRegex(LedgerError, "already open"):
                ledger.open_allocation()
            with self.assertRaises(ValueError):
                Ledger(Path(directory) / "other.jsonl", cap_hours=51)
            with self.assertRaises(ValueError):
                Ledger(Path(directory) / "other.jsonl", cap_hours=0)

    def test_time_travel_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            clock = FakeClock()
            ledger = Ledger(path, clock=clock)
            ledger.open_allocation()
            clock.advance(-10)
            with self.assertRaisesRegex(LedgerError, "backwards"):
                ledger.consumed_hours()


if __name__ == "__main__":
    unittest.main()
