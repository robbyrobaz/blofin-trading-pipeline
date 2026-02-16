#!/bin/bash
# Install systemd timer for blofin-stack daily pipeline

set -e

WORKSPACE="$HOME/.openclaw/workspace/blofin-stack"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

echo "Installing blofin-stack daily pipeline timer..."

# Create systemd user directory if it doesn't exist
mkdir -p "$SYSTEMD_USER_DIR"

# Copy service and timer files
cp "$WORKSPACE/blofin-stack-daily.service" "$SYSTEMD_USER_DIR/"
cp "$WORKSPACE/blofin-stack-daily.timer" "$SYSTEMD_USER_DIR/"

echo "Files copied to $SYSTEMD_USER_DIR"

# Reload systemd
systemctl --user daemon-reload

# Enable and start timer
systemctl --user enable blofin-stack-daily.timer
systemctl --user start blofin-stack-daily.timer

echo ""
echo "✓ Installation complete!"
echo ""
echo "Timer status:"
systemctl --user status blofin-stack-daily.timer --no-pager
echo ""
echo "Next scheduled run:"
systemctl --user list-timers blofin-stack-daily.timer --no-pager
echo ""
echo "Useful commands:"
echo "  - Check timer status: systemctl --user status blofin-stack-daily.timer"
echo "  - View logs: journalctl --user -u blofin-stack-daily.service -f"
echo "  - Run manually: systemctl --user start blofin-stack-daily.service"
echo "  - Stop timer: systemctl --user stop blofin-stack-daily.timer"
echo "  - Disable timer: systemctl --user disable blofin-stack-daily.timer"
