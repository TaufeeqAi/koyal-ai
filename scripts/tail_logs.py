"""
Convenient log viewer for KoyalAI.

Usage:
    python scripts/tail_logs.py              # Show last 100 lines
    python scripts/tail_logs.py -f           # Follow (like tail -f)
    python scripts/tail_logs.py -n 50        # Show last 50 lines
    python scripts/tail_logs.py --grep STT   # Filter for STT lines
    python scripts/tail_logs.py --grep "greeting|apology|pipeline"  # Regex
"""

import argparse
import re
import sys
import time
from pathlib import Path

LOG_FILE = Path("logs/koyalai.log")


def tail_file(filepath: Path, n: int = 100):
    """Return last N lines of a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return lines[-n:]
    except FileNotFoundError:
        print(f"Log file not found: {filepath}")
        print("Start the backend first to generate logs.")
        sys.exit(1)


def follow_file(filepath: Path, pattern: re.Pattern | None = None):
    """Follow a file like tail -f, optionally filtering."""
    print(f"Following {filepath} (Ctrl+C to stop)...")
    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(0, 2)  # Jump to end
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            if pattern is None or pattern.search(line):
                print(line, end="")


def main():
    parser = argparse.ArgumentParser(description="View KoyalAI logs")
    parser.add_argument("-n", "--lines", type=int, default=100, help="Number of lines to show")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    parser.add_argument("--grep", type=str, help="Filter lines by regex pattern")
    args = parser.parse_args()

    pattern = re.compile(args.grep, re.IGNORECASE) if args.grep else None

    if args.follow:
        follow_file(LOG_FILE, pattern)
    else:
        lines = tail_file(LOG_FILE, args.lines)
        for line in lines:
            if pattern is None or pattern.search(line):
                print(line, end="")


if __name__ == "__main__":
    main()