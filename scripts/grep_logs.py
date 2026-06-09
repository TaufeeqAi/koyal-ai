"""
Search KoyalAI logs for specific events.

Usage:
    python scripts/grep_logs.py greeting       # Find greeting-related logs
    python scripts/grep_logs.py STT            # Find STT logs
    python scripts/grep_logs.py pipeline       # Find pipeline logs
    python scripts/grep_logs.py error          # Find errors
    python scripts/grep_logs.py "track_subscribed|track_published"  # Regex
"""

import argparse
import re
import sys
from pathlib import Path

LOG_FILE = Path("logs/koyalai.log")


def grep_logs(pattern: str, context: int = 2):
    """Search logs with context lines."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Log file not found: {LOG_FILE}")
        sys.exit(1)

    regex = re.compile(pattern, re.IGNORECASE)
    matches = []

    for i, line in enumerate(lines):
        if regex.search(line):
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            matches.append((start, end, i))

    if not matches:
        print(f'No matches for "{pattern}"')
        return

    print(f"Found {len(matches)} match(es) for '{pattern}'\n")

    last_end = -1
    for start, end, match_idx in matches:
        if start <= last_end:
            start = last_end + 1
        if start >= end:
            continue

        for j in range(start, end):
            prefix = ">>> " if j == match_idx else "    "
            print(f"{prefix}{lines[j]}", end="")
        print("─" * 80)
        last_end = end - 1


def main():
    parser = argparse.ArgumentParser(description="Search KoyalAI logs")
    parser.add_argument("pattern", help="Search pattern (regex supported)")
    parser.add_argument("-C", "--context", type=int, default=2, help="Context lines")
    args = parser.parse_args()

    grep_logs(args.pattern, args.context)


if __name__ == "__main__":
    main()