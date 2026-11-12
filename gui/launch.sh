#!/usr/bin/env bash
# Start the AAC hand demo GUI with the hand environment, from anywhere (the desktop icon runs this). Its output, which
# a terminal would show, goes to ~/.local/state/artly-os/gui.log (the previous run's to gui.log.1).
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/artly-os"
mkdir -p "$LOG_DIR"
[[ -f $LOG_DIR/gui.log ]] && mv -f "$LOG_DIR/gui.log" "$LOG_DIR/gui.log.1"
if [[ ! -x hand/.hand/bin/python ]]; then
    echo "hand/.hand is missing: run ./setup_envs.sh hand in $(pwd)" | tee "$LOG_DIR/gui.log" >&2
    command -v notify-send >/dev/null && notify-send "Artly-os" "Not set up yet: run ./setup_envs.sh hand in $(pwd)"
    exit 1
fi
echo "[launch] $(date '+%F %T') $(pwd)" >"$LOG_DIR/gui.log"
exec hand/.hand/bin/python -u gui/main.py >>"$LOG_DIR/gui.log" 2>&1
