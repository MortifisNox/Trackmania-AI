#!/usr/bin/env bash
# One-click start on Linux: ./start_training.sh trains with the settings in
# tmai_settings.json. Updates the code if asked, then hands over to
# `tmai start`, which installs the plugins and starts the dashboard and training.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root" || exit 1

python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
    echo "No .venv found in $root. Do docs/SETUP_LINUX.md first." >&2
    exit 1
fi

settings="$root/tmai_settings.json"
if [ ! -f "$settings" ]; then
    cp "$root/tmai_settings.example.json" "$settings"
    echo "Created tmai_settings.json. Write your map name and buttons in it, save, then run ./start_training.sh again."
    "${VISUAL:-${EDITOR:-xdg-open}}" "$settings" >/dev/null 2>&1 &
    exit 1
fi

# update_code defaults to true, like tmai's own default.
update="$("$python" -c 'import json,sys; s=json.load(open(sys.argv[1], encoding="utf-8-sig")); print(str(s.get("update_code", True)).lower())' "$settings" 2>&1)" || {
    echo "tmai_settings.json is not valid JSON: $update" >&2
    exit 1
}

if [ "$update" = "true" ]; then
    if ! command -v git >/dev/null 2>&1; then
        echo "git not found; starting with the code you have."
    else
        echo "Updating the code (git pull)..."
        before="$(git rev-parse HEAD)"
        if ! git pull --ff-only; then
            echo "git pull failed; starting with the code you have. See docs/TROUBLESHOOTING.md."
        elif [ "$(git rev-parse HEAD)" != "$before" ]; then
            echo "New code pulled; reinstalling the Python package..."
            "$python" -m pip install --quiet -e ".[game,dashboard,dev]" || {
                echo "pip install failed; see the messages above." >&2
                exit 1
            }
        fi
    fi
fi

exec "$python" -m tmai start
