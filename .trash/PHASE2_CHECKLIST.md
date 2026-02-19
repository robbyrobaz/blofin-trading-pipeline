# Phase 2 ML Retrain Framework — Deployment Checklist

**Build Date:** 2026-02-17
**Build Status:** ✅ COMPLETE

## Pre-Deployment Verification

### Code Modules
- [x] `ml_retrain_phase2.py` — 28.8 KB
  - [x] Gate checks (age, trade count, regime diversity)
  - [x] Efficient tick sampling (solves 28M-row bottleneck)
  - [x] Feature engineering (11 features)
  - [x] Model training (5 GradientBoosting models)
  - [x] Walk-forward split with 24h embargo
  - [x] Database logging
  - [x] CLI: `--dry-run`, `--force`, `--smoke-test`, `--db PATH`

- [x] `ab_test_models.py` — 17.4 KB
  - [x] Trade arm splitting (v1_backtest vs v2_paper)
  - [x] Sharpe ratio comparison
  - [x] Expectancy gate (must be > 0)
  - [x] Automatic model swap
  - [x] Full audit logging
  - [x] CLI: `--dry-run`, `--status`, `--force-swap`, `--db PATH`

- [x] `execution_calibrator_v2.py` — 20.7 KB
  - [x] EMA-weighted slippage (recent 50 trades 3× boost)
  - [x] Time-of-day bucketing (4 UTC windows)
  - [x] Volatility regime classification (trending/ranging/volatile)
  - [x] Position size multipliers (0.5x–1.2x)
  - [x] Backward-compatibility alias (avg_slippage_per_side_pct)
  - [x] Comprehensive recommendations

- [x] `execution_calibrator.py` — PATCHED
  - [x] Added backward-compat field alias
  - [x] No breaking changes

### Testing
- [x] `smoke_test_phase2.py` — 15.1 KB
  - [x] Gate checks test
  - [x] Feature building test
  - [x] A/B evaluation test
  - [x] Execution calibrator test
  - [x] Database integrity test
  - [x] File outputs test
  - [x] **Result: 6/6 checks PASSED** ✅

### Scheduling
- [x] `blofin-stack-phase2-retrain.service` — 684 B
  - [x] Runs at 06:00 MST (13:00 UTC daily)
  - [x] Timeout: 2h, Memory: 4GB, CPU: 300%

- [x] `blofin-stack-phase2-retrain.timer` — 333 B
  - [x] OnCalendar set to 13:00 UTC (06:00 MST)
  - [x] Persistent=true, RandomizedDelaySec=60

- [x] `blofin-stack-phase2-ab-eval.service` — 641 B
  - [x] Runs at 18:00 MST (01:00 UTC daily)
  - [x] Timeout: 30m, Memory: 2GB, CPU: 200%

- [x] `blofin-stack-phase2-ab-eval.timer` — 349 B
  - [x] OnCalendar set to 01:00 UTC (18:00 MST previous day)
  - [x] Persistent=true, RandomizedDelaySec=60

- [x] `cron_phase2_check.sh` — 1002 B (already present, executable)
  - [x] Runs `ml_retrain_phase2.py`
  - [x] Logs to `logs/phase2_retrain.log`

- [x] `cron_ab_evaluate.sh` — 1002 B (already present, executable)
  - [x] Runs `ab_test_models.py`
  - [x] Logs to same file

### Documentation
- [x] `PHASE2_DEPLOYMENT.md` — 13.9 KB
  - [x] Architecture overview with diagrams
  - [x] Full component inventory
  - [x] Installation instructions (systemd + cron)
  - [x] Gate logic explanation
  - [x] Model training details
  - [x] A/B testing workflow
  - [x] Execution calibration specs
  - [x] Output file formats
  - [x] Troubleshooting guide
  - [x] Monitoring checklist
  - [x] Performance tuning
  - [x] Integration points
  - [x] Rollback procedure

- [x] `PHASE2_BUILD_SUMMARY.md` — 12.6 KB
  - [x] Deliverables summary
  - [x] Technical highlights
  - [x] Code quality notes
  - [x] Database schema changes
  - [x] Deployment quick-start
  - [x] Timeline & execution flow
  - [x] Known limitations
  - [x] Sign-off

- [x] `PHASE2_CHECKLIST.md` — This file
  - [x] Complete verification checklist

## Verification Commands

```bash
# 1. Verify all modules import correctly
cd /home/rob/.openclaw/workspace/blofin-stack
.venv/bin/python3 -c "
from ml_retrain_phase2 import run_phase2_retrain
from ab_test_models import run_ab_evaluation
from execution_calibrator_v2 import compute_calibration
print('✅ All modules import successfully')
"

# 2. Run comprehensive smoke test
.venv/bin/python3 smoke_test_phase2.py
# Expected output: "6/6 checks passed"

# 3. Verify systemd files are valid
systemd-analyze verify blofin-stack-phase2-retrain.service
systemd-analyze verify blofin-stack-phase2-ab-eval.service

# 4. Test dry-run (gates check only)
.venv/bin/python3 ml_retrain_phase2.py --dry-run

# 5. Check A/B status
.venv/bin/python3 ab_test_models.py --status

# 6. Run calibrator
.venv/bin/python3 execution_calibrator_v2.py

# 7. Verify database schema
sqlite3 data/blofin_monitor.db ".tables" | grep phase2
sqlite3 data/blofin_monitor.db ".tables" | grep ab_
```

## Installation Instructions

### Option A: Systemd (Recommended for production)

```bash
# 1. Ensure user-level systemd is running
systemctl --user status >/dev/null || systemctl --user daemon

# 2. Copy service files
mkdir -p ~/.config/systemd/user
cp blofin-stack-phase2-*.service ~/.config/systemd/user/
cp blofin-stack-phase2-*.timer ~/.config/systemd/user/

# 3. Reload and enable
systemctl --user daemon-reload
systemctl --user enable blofin-stack-phase2-retrain.timer
systemctl --user enable blofin-stack-phase2-ab-eval.timer

# 4. Start
systemctl --user start blofin-stack-phase2-retrain.timer
systemctl --user start blofin-stack-phase2-ab-eval.timer

# 5. Verify
systemctl --user status blofin-stack-phase2-retrain.timer
systemctl --user status blofin-stack-phase2-ab-eval.timer
systemctl --user list-timers
```

### Option B: Traditional Cron

```bash
# Edit crontab
crontab -e

# Add these lines:
# Phase 2 Retrain — Daily @ 06:00 MST (13:00 UTC)
0 13 * * * /home/rob/.openclaw/workspace/blofin-stack/cron_phase2_check.sh

# A/B Evaluation — Daily @ 18:00 MST (01:00 UTC next day)
0 1 * * * /home/rob/.openclaw/workspace/blofin-stack/cron_ab_evaluate.sh
```

## Post-Installation Verification

- [ ] Systemd timers enabled: `systemctl --user status blofin-stack-phase2-*.timer`
- [ ] Cron jobs added (if using cron): `crontab -l | grep phase2`
- [ ] Log file created: `ls -la logs/phase2_retrain.log`
- [ ] Database tables exist:
  ```bash
  sqlite3 data/blofin_monitor.db \
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;" | \
    grep -E 'phase2|ab_'
  ```
- [ ] Calibration file exists: `ls -la data/execution_calibration.json`

## First Run Checklist

**Day 1 (Installation):**
- [ ] Install systemd or cron per instructions above
- [ ] Verify timers/jobs are active
- [ ] Monitor first execution at scheduled time
- [ ] Check logs: `tail -f logs/phase2_retrain.log`

**Day 2 (After first Phase 2 trigger):**
- [ ] Check if retrain was attempted
- [ ] Review gate logs: `phase2_gate_log` table
- [ ] Verify `phase2_retrain_runs` has a record

**Day 3 (After A/B evaluation):**
- [ ] Check `ab_test_runs` for evaluation record
- [ ] Verify A/B decision logged
- [ ] Check if model swap was executed (if conditions met)

## Success Criteria

Phase 2 is **successfully deployed** when ALL of the following are true:

1. [ ] Smoke test passes (6/6 checks)
   ```bash
   .venv/bin/python3 smoke_test_phase2.py
   # Expected: "Result: 6/6 checks passed"
   ```

2. [ ] Systemd timers active (or cron jobs installed)
   ```bash
   systemctl --user list-timers | grep phase2
   # Or: crontab -l | grep phase2
   ```

3. [ ] No errors in logs for 24 hours
   ```bash
   tail -50 logs/phase2_retrain.log | grep -i error
   # Should return nothing or only INFO level
   ```

4. [ ] Database tables populated
   ```bash
   sqlite3 data/blofin_monitor.db \
     "SELECT COUNT(*) FROM phase2_gate_log;"
   # Should have > 0 rows after first run
   ```

5. [ ] Execution calibration updated daily
   ```bash
   ls -la data/execution_calibration.json
   # modified time should be recent
   ```

6. [ ] A/B test records appear
   ```bash
   sqlite3 data/blofin_monitor.db \
     "SELECT COUNT(*) FROM ab_test_runs;"
   # Should have > 0 rows after 2nd day
   ```

## Monitoring Dashboard

### Daily Checks

```bash
#!/bin/bash
# Save as: scripts/phase2_status.sh

echo "=== Phase 2 Status ==="
echo
echo "Last 5 Phase 2 runs:"
sqlite3 ~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db \
  "SELECT ts_iso, gate_pass, success, duration_sec FROM phase2_retrain_runs ORDER BY ts_ms DESC LIMIT 5;"

echo
echo "Last 5 A/B evaluations:"
sqlite3 ~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db \
  "SELECT ts_iso, arm_a_trades, arm_b_trades, swap_recommended, swap_executed FROM ab_test_runs ORDER BY ts_ms DESC LIMIT 5;"

echo
echo "Current active model:"
sqlite3 ~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db \
  "SELECT active_arm, swapped_at FROM ab_active_model WHERE id=1;"

echo
echo "Last execution calibration:"
ls -lh ~/.openclaw/workspace/blofin-stack/data/execution_calibration.json
```

## Rollback Plan

If issues occur after deployment:

```bash
# Disable timers (keep files installed)
systemctl --user disable blofin-stack-phase2-retrain.timer
systemctl --user disable blofin-stack-phase2-ab-eval.timer

# Or remove cron lines:
crontab -e
# Delete the two Phase 2 lines

# Check logs for error details
tail -100 logs/phase2_retrain.log | grep ERROR

# Revert to last known-good model if needed
# (backed up in data/models/*_v2_paper_prev/)
```

## File Inventory

**Core Modules** (Ready)
```
✅ ml_retrain_phase2.py                 (28.8 KB)
✅ ab_test_models.py                    (17.4 KB)
✅ execution_calibrator_v2.py           (20.7 KB)
✅ execution_calibrator.py              (patched for backward-compat)
```

**Testing** (Ready)
```
✅ smoke_test_phase2.py                 (15.1 KB)
✅ logs/smoke_test_phase2_results.log   (test output)
```

**Scheduling** (Ready)
```
✅ blofin-stack-phase2-retrain.service  (684 B)
✅ blofin-stack-phase2-retrain.timer    (333 B)
✅ blofin-stack-phase2-ab-eval.service  (641 B)
✅ blofin-stack-phase2-ab-eval.timer    (349 B)
✅ cron_phase2_check.sh                 (1002 B, executable)
✅ cron_ab_evaluate.sh                  (1002 B, executable)
```

**Documentation** (Ready)
```
✅ PHASE2_DEPLOYMENT.md                 (13.9 KB)
✅ PHASE2_BUILD_SUMMARY.md              (12.6 KB)
✅ PHASE2_CHECKLIST.md                  (this file)
```

**Database** (Auto-created)
```
Auto → phase2_retrain_runs              (audit trail)
Auto → phase2_gate_log                  (gate decisions)
Auto → ab_test_runs                     (A/B results)
Auto → ab_active_model                  (current active model)
```

**Outputs** (Auto-created)
```
Auto → data/models/*_v2_paper/          (new model artifacts)
Auto → data/models/*_v2_paper_prev/     (previous backups)
Auto → data/execution_calibration.json  (daily calibration)
Auto → logs/phase2_retrain.log          (shared log)
```

## Support Resources

| Resource | Location | Purpose |
|----------|----------|---------|
| Deployment guide | `PHASE2_DEPLOYMENT.md` | Full setup & configuration |
| Build summary | `PHASE2_BUILD_SUMMARY.md` | Architecture & design |
| This checklist | `PHASE2_CHECKLIST.md` | Verification & installation |
| Main log file | `logs/phase2_retrain.log` | Daily execution logs |
| Smoke test log | `logs/smoke_test_phase2_results.log` | Test verification |
| Database queries | SQLite CLI | `data/blofin_monitor.db` |

## Next Steps

1. **Install per instructions above** (systemd recommended)
2. **Run verification commands** to confirm installation
3. **Wait for first scheduled execution** (next 06:00 MST for retrain, then 18:00 MST for A/B)
4. **Review logs** for any errors or warnings
5. **Monitor success criteria** (see above checklist)
6. **Adjust gates/parameters** if needed after observing behavior

---

**Ready for Production Deployment: ✅ YES**

**Build verified:** 2026-02-17 18:19 UTC
**Smoke test result:** 6/6 passed
**Code review:** Complete
**Documentation:** Complete
