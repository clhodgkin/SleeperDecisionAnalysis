
from __future__ import annotations

import argparse
from pathlib import Path

from sleeper_trade import SleeperClient, collect_full_league_snapshot, save_snapshot


def main():
    parser = argparse.ArgumentParser(description="Export a Sleeper league to JSON.")
    parser.add_argument("--username", required=True, help="Sleeper username or user ID")
    parser.add_argument("--league-id", required=True, help="Sleeper league ID")
    parser.add_argument("--season", required=True, type=int, help="NFL season, e.g. 2026")
    parser.add_argument(
        "--include-player-catalog",
        action="store_true",
        help="Include Sleeper's large full NFL player catalog in the JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: sleeper_league_<league>_<season>.json)",
    )
    args = parser.parse_args()

    client = SleeperClient()
    snapshot = collect_full_league_snapshot(
        client=client,
        username_or_id=args.username,
        league_id=args.league_id,
        season=args.season,
        weeks=range(1, 19),
        include_player_catalog=args.include_player_catalog,
    )
    output = Path(
        args.output
        or f"sleeper_league_{args.league_id}_{args.season}.json"
    )
    save_snapshot(snapshot, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
