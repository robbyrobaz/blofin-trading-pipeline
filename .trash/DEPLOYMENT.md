# Deployment Guide - Systemd Timer Setup

This guide explains how the daily AI pipeline is deployed to run automatically.

## Overview

The pipeline runs daily at 00:00 UTC via systemd user timer.

Components:
- `blofin-stack-daily.service` - Systemd service unit
- `blofin-stack-daily.timer` - Systemd timer unit
- `/usr/local/bin/blofin-ai-pipeline` - Wrapper script

## Installation

### 1. Install the Wrapper Script

```bash
cd /home/rob/.openclaw/workspace/blofin-stack

# Copy wrapper script to /usr/local/bin (requires sudo)
sudo cp blofin-ai-pipeline /usr/local/bin/
sudo chmod +x /usr/local/bin/blofin-ai-pipeline
```

### 2. Install Systemd Units

```bash
# Copy service and timer to user systemd directory
mkdir -p ~/.config/systemd/user
cp blofin-stack-daily.service ~/.config/systemd/user/
cp blofin-stack-daily.timer ~/.config/systemd/user/

# Reload systemd
systemctl --user daemon-reload
```

### 3. Enable and Start the Timer

```bash
# Enable timer (starts automatically on boot)
systemctl --user enable blofin-stack-daily.timer

# Start timer immediately
systemctl --user start blofin-stack-daily.timer

# Verify timer is active
systemctl --user status blofin-stack-daily.timer
```

## Verification

Check timer status:
```bash
systemctl --user list-timers
```

You should see:
```
NEXT                         LEFT          LAST PASSED UNIT                       ACTIVATES
Tue 2026-02-16 00:00:00 MST  2h 15min left n/a  n/a    blofin-stack-daily.timer   blofin-stack-daily.service
```

## Manual Execution

To run the pipeline manually (outside the timer):

```bash
# Via wrapper script
/usr/local/bin/blofin-ai-pipeline

# Or directly
cd /home/rob/.openclaw/workspace/blofin-stack
source .venv/bin/activate
python orchestration/daily_runner.py
```

## Monitoring

### Check Logs

```bash
# View pipeline log
tail -f /home/rob/.openclaw/workspace/blofin-stack/data/pipeline.log

# View systemd journal
journalctl --user -u blofin-stack-daily.service -f
```

### Check Last Run

```bash
systemctl --user status blofin-stack-daily.service
```

Output shows:
- Last execution time
- Exit status (0 = success)
- Recent log lines

## Configuration

### Change Schedule

Edit `~/.config/systemd/user/blofin-stack-daily.timer`:

```ini
[Timer]
# Run daily at 00:00 UTC
OnCalendar=daily

# Or specify exact time:
# OnCalendar=*-*-* 03:00:00  # 3 AM every day
# OnCalendar=Mon *-*-* 00:00:00  # Mondays only
```

After editing:
```bash
systemctl --user daemon-reload
systemctl --user restart blofin-stack-daily.timer
```

### Timeout and Retry

The service will:
- Restart on failure (after 60 seconds)
- Log to `data/pipeline.log`
- Exit with status 1 on error

## Disabling

To stop automatic execution:

```bash
# Stop timer
systemctl --user stop blofin-stack-daily.timer

# Disable (won't start on boot)
systemctl --user disable blofin-stack-daily.timer
```

## Uninstallation

```bash
# Stop and disable
systemctl --user stop blofin-stack-daily.timer
systemctl --user disable blofin-stack-daily.timer

# Remove files
rm ~/.config/systemd/user/blofin-stack-daily.{service,timer}
sudo rm /usr/local/bin/blofin-ai-pipeline

# Reload
systemctl --user daemon-reload
```

## Troubleshooting

### Timer Not Triggering

1. Check timer is active: `systemctl --user is-active blofin-stack-daily.timer`
2. Check timer list: `systemctl --user list-timers`
3. View logs: `journalctl --user -u blofin-stack-daily.timer`

### Service Failing

1. Check service status: `systemctl --user status blofin-stack-daily.service`
2. View full logs: `journalctl --user -u blofin-stack-daily.service -n 50`
3. Test wrapper script manually: `/usr/local/bin/blofin-ai-pipeline`

### Permission Issues

Ensure:
- Wrapper script is executable: `chmod +x /usr/local/bin/blofin-ai-pipeline`
- Virtual environment exists: `ls /home/rob/.openclaw/workspace/blofin-stack/.venv`
- Database is writable: `ls -l data/blofin_monitor.db`

## Alternative: Cron

If systemd user timers don't work, use cron instead:

```bash
crontab -e
```

Add:
```
# Run daily at 00:00 UTC
0 0 * * * /usr/local/bin/blofin-ai-pipeline >> /home/rob/.openclaw/workspace/blofin-stack/data/cron.log 2>&1
```

Save and exit. Verify:
```bash
crontab -l
```
