
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from scipy.optimize import linear_sum_assignment


API_BASE = "https://api.sleeper.app/v1"
PROJ_BASE = "https://api.sleeper.com"

STARTER_SLOT_ELIGIBILITY = {
    "QB": {"QB"},
    "RB": {"RB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "K": {"K"},
    "DEF": {"DEF"},
    "DL": {"DL"},
    "LB": {"LB"},
    "DB": {"DB"},
    "FLEX": {"RB", "WR", "TE"},
    "WRRB_FLEX": {"WR", "RB"},
    "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "IDP_FLEX": {"DL", "LB", "DB"},
}

NON_STARTER_SLOTS = {"BN", "IR", "TAXI"}


class SleeperAPIError(RuntimeError):
    pass


class SleeperClient:
    """
    Thin client around Sleeper's public read-only API.

    The documented league/user endpoints live on api.sleeper.app/v1.
    Sleeper's projections feed is currently available on api.sleeper.com,
    but it is not part of the current official Sleeper API documentation.
    """

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/151.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
            }
        )

    def _get(self, url: str, params: dict | None = None) -> Any:
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise SleeperAPIError(f"Request failed: {url}\n{exc}") from exc
        except ValueError as exc:
            raise SleeperAPIError(f"Sleeper returned non-JSON data: {url}") from exc

    def get_user(self, username_or_id: str) -> dict:
        return self._get(f"{API_BASE}/user/{username_or_id}")

    def get_nfl_state(self) -> dict:
        return self._get(f"{API_BASE}/state/nfl")

    def get_leagues(self, user_id: str, season: int | str) -> list[dict]:
        return self._get(f"{API_BASE}/user/{user_id}/leagues/nfl/{season}")

    def get_league(self, league_id: str) -> dict:
        return self._get(f"{API_BASE}/league/{league_id}")

    def get_rosters(self, league_id: str) -> list[dict]:
        return self._get(f"{API_BASE}/league/{league_id}/rosters")

    def get_league_users(self, league_id: str) -> list[dict]:
        return self._get(f"{API_BASE}/league/{league_id}/users")

    def get_matchups(self, league_id: str, week: int) -> list[dict]:
        return self._get(f"{API_BASE}/league/{league_id}/matchups/{week}")

    def get_transactions(self, league_id: str, week: int) -> list[dict]:
        return self._get(f"{API_BASE}/league/{league_id}/transactions/{week}")

    def get_traded_picks(self, league_id: str) -> list[dict]:
        return self._get(f"{API_BASE}/league/{league_id}/traded_picks")

    def get_winners_bracket(self, league_id: str) -> list[dict]:
        return self._get(f"{API_BASE}/league/{league_id}/winners_bracket")

    def get_losers_bracket(self, league_id: str) -> list[dict]:
        # Sleeper's endpoint is historically spelled "losers_bracket".
        return self._get(f"{API_BASE}/league/{league_id}/losers_bracket")

    def get_drafts(self, league_id: str) -> list[dict]:
        return self._get(f"{API_BASE}/league/{league_id}/drafts")

    def get_draft(self, draft_id: str) -> dict:
        return self._get(f"{API_BASE}/draft/{draft_id}")

    def get_draft_picks(self, draft_id: str) -> list[dict]:
        return self._get(f"{API_BASE}/draft/{draft_id}/picks")

    def get_draft_traded_picks(self, draft_id: str) -> list[dict]:
        return self._get(f"{API_BASE}/draft/{draft_id}/traded_picks")

    def get_players(self) -> dict:
        # Large response. Cache it in the caller rather than repeatedly fetching.
        return self._get(f"{API_BASE}/players/nfl")

    def get_week_projections(self, season: int | str, week: int) -> Any:
        """
        Fetch Sleeper's currently available weekly projection feed.

        This endpoint is unofficial/undocumented, so response normalization is
        deliberately handled elsewhere rather than coupled to one JSON shape.
        """
        params = [
            ("season_type", "regular"),
            ("position[]", "QB"),
            ("position[]", "RB"),
            ("position[]", "WR"),
            ("position[]", "TE"),
            ("position[]", "K"),
            ("position[]", "DEF"),
            ("order_by", "pts_ppr"),
        ]
        primary = f"{PROJ_BASE}/projections/nfl/{season}/{week}"
        try:
            return self._get(primary, params=params)
        except SleeperAPIError:
            # Older community wrappers have used this v1 path. Keep it only
            # as a fallback so the app is less brittle.
            fallback = f"{API_BASE}/projections/nfl/regular/{season}/{week}"
            return self._get(fallback)


def _safe_get(d: dict | None, key: str, default: Any = None) -> Any:
    return (d or {}).get(key, default)


def normalize_projection_payload(payload: Any) -> dict[str, dict]:
    """
    Normalize known Sleeper projection response shapes into:
        player_id -> {
            "player_id": str,
            "stats": dict,
            "player": dict,
            "raw": dict,
        }

    Handles:
      * list[record] where record has player_id/player/stats
      * dict[player_id] -> stats/record
    """
    out: dict[str, dict] = {}

    if isinstance(payload, list):
        rows = payload
        for row in rows:
            if not isinstance(row, dict):
                continue
            player = row.get("player") or {}
            pid = row.get("player_id") or player.get("player_id")
            if pid is None:
                continue
            stats = row.get("stats") if isinstance(row.get("stats"), dict) else row
            out[str(pid)] = {
                "player_id": str(pid),
                "stats": stats or {},
                "player": player,
                "raw": row,
            }
        return out

    if isinstance(payload, dict):
        for pid, row in payload.items():
            if not isinstance(row, dict):
                continue
            player = row.get("player") or {}
            actual_pid = row.get("player_id") or player.get("player_id") or pid
            stats = row.get("stats") if isinstance(row.get("stats"), dict) else row
            out[str(actual_pid)] = {
                "player_id": str(actual_pid),
                "stats": stats or {},
                "player": player,
                "raw": row,
            }

    return out


def scoring_key_from_league(league: dict) -> str:
    """
    Pick Sleeper's closest standard projection rollup from reception scoring.
    This is intentionally simple and transparent.
    """
    rec = float((league.get("scoring_settings") or {}).get("rec", 0.0) or 0.0)
    if abs(rec - 1.0) < 0.05:
        return "pts_ppr"
    if abs(rec - 0.5) < 0.05:
        return "pts_half_ppr"
    if abs(rec) < 0.05:
        return "pts_std"
    # For unusual reception scoring, half-PPR is just a fallback display basis.
    return "pts_half_ppr"


def projection_points(
    normalized_record: dict,
    score_key: str,
    league_scoring: dict | None = None,
    use_custom_scoring: bool = False,
) -> float:
    stats = normalized_record.get("stats") or {}
    raw = normalized_record.get("raw") or {}

    if use_custom_scoring and league_scoring:
        # Most Sleeper scoring settings use the same stat keys as the stat/projection
        # payload. Dot-product the overlapping keys. This handles many custom leagues,
        # but exotic threshold/position-premium bonuses may still need custom logic.
        total = 0.0
        matched = 0
        for stat_name, multiplier in league_scoring.items():
            if stat_name in stats:
                try:
                    total += float(stats.get(stat_name, 0) or 0) * float(multiplier or 0)
                    matched += 1
                except (TypeError, ValueError):
                    pass
        if matched:
            return float(total)

    for source in (stats, raw):
        val = source.get(score_key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def projection_position(normalized_record: dict, players: dict, player_id: str) -> set[str]:
    player = normalized_record.get("player") or {}
    master = players.get(player_id) or {}

    positions = (
        master.get("fantasy_positions")
        or player.get("fantasy_positions")
        or []
    )
    if isinstance(positions, str):
        positions = [positions]

    position = master.get("position") or player.get("position")
    if position and position not in positions:
        positions = list(positions) + [position]

    # Team-defense IDs are commonly abbreviations such as BUF, PHI, etc.
    if not positions and player_id.isalpha() and len(player_id) <= 4:
        positions = ["DEF"]

    return {str(p) for p in positions if p}


def starter_slots(league: dict) -> list[str]:
    return [
        slot for slot in (league.get("roster_positions") or [])
        if slot not in NON_STARTER_SLOTS
    ]


def eligible_for_slot(player_positions: set[str], slot: str) -> bool:
    allowed = STARTER_SLOT_ELIGIBILITY.get(slot)
    if allowed is None:
        # Unknown custom slot: require an exact positional match.
        return slot in player_positions
    return bool(player_positions & allowed)


def optimal_lineup(
    roster_player_ids: Iterable[str],
    slots: list[str],
    points: dict[str, float],
    position_map: dict[str, set[str]],
) -> tuple[float, list[tuple[str, str, float]]]:
    """
    Maximum-weight assignment of players to starting slots.

    Dummy columns allow empty slots and keep the solver feasible when a roster
    has no eligible player for a slot.
    """
    player_ids = list(dict.fromkeys(str(p) for p in roster_player_ids if p))
    n_slots = len(slots)
    if n_slots == 0:
        return 0.0, []

    # Add one dummy player per slot, each worth 0.
    n_cols = len(player_ids) + n_slots
    weights = np.zeros((n_slots, n_cols), dtype=float)
    invalid = -1_000_000.0
    if player_ids:
        weights[:, : len(player_ids)] = invalid

    for r, slot in enumerate(slots):
        for c, pid in enumerate(player_ids):
            if eligible_for_slot(position_map.get(pid, set()), slot):
                weights[r, c] = float(points.get(pid, 0.0) or 0.0)

    rows, cols = linear_sum_assignment(weights, maximize=True)
    lineup = []
    total = 0.0
    for r, c in zip(rows, cols):
        if c < len(player_ids) and weights[r, c] > invalid / 2:
            pid = player_ids[c]
            val = max(0.0, float(weights[r, c]))
            total += val
            lineup.append((slots[r], pid, val))
    return total, lineup


def build_team_maps(users: list[dict], rosters: list[dict]) -> tuple[dict[int, dict], dict[str, int]]:
    users_by_id = {str(u.get("user_id")): u for u in users}
    teams: dict[int, dict] = {}
    owner_to_roster: dict[str, int] = {}

    for roster in rosters:
        rid = int(roster["roster_id"])
        owner_id = str(roster.get("owner_id")) if roster.get("owner_id") is not None else ""
        user = users_by_id.get(owner_id, {})
        metadata = user.get("metadata") or {}
        team_name = (
            metadata.get("team_name")
            or user.get("display_name")
            or user.get("username")
            or f"Roster {rid}"
        )
        teams[rid] = {
            "roster_id": rid,
            "owner_id": owner_id,
            "team_name": team_name,
            "username": user.get("username"),
            "display_name": user.get("display_name"),
            "players": [str(p) for p in (roster.get("players") or [])],
            "starters": [str(p) for p in (roster.get("starters") or [])],
            "reserve": [str(p) for p in (roster.get("reserve") or [])],
            "settings": roster.get("settings") or {},
        }
        if owner_id:
            owner_to_roster[owner_id] = rid

    return teams, owner_to_roster


def player_display_name(player_id: str, players: dict) -> str:
    p = players.get(str(player_id)) or {}
    return (
        p.get("full_name")
        or " ".join(x for x in [p.get("first_name"), p.get("last_name")] if x)
        or p.get("team")
        or str(player_id)
    )


def collect_projection_data(
    client: SleeperClient,
    season: int,
    weeks: Iterable[int],
    players: dict,
    league: dict,
    scoring_mode: str = "auto",
) -> tuple[dict[int, dict[str, float]], dict[str, set[str]], str, list[int]]:
    """
    Returns:
      weekly_points[week][player_id] = projected points
      position_map[player_id] = eligible fantasy positions
      score_key used
      weeks that returned projection data
    """
    if scoring_mode == "ppr":
        score_key = "pts_ppr"
    elif scoring_mode == "half_ppr":
        score_key = "pts_half_ppr"
    elif scoring_mode == "standard":
        score_key = "pts_std"
    else:
        score_key = scoring_key_from_league(league)

    use_custom = scoring_mode == "custom"
    weekly_points: dict[int, dict[str, float]] = {}
    position_map: dict[str, set[str]] = {}
    good_weeks: list[int] = []

    for week in weeks:
        payload = client.get_week_projections(season, int(week))
        normalized = normalize_projection_payload(payload)
        if not normalized:
            continue

        good_weeks.append(int(week))
        week_points: dict[str, float] = {}
        for pid, record in normalized.items():
            week_points[pid] = projection_points(
                record,
                score_key=score_key,
                league_scoring=league.get("scoring_settings") or {},
                use_custom_scoring=use_custom,
            )
            position_map[pid] = projection_position(record, players, pid)
        weekly_points[int(week)] = week_points

    # Ensure rostered players have positions even if no projection row exists.
    for pid, p in players.items():
        if pid not in position_map:
            positions = p.get("fantasy_positions") or []
            if isinstance(positions, str):
                positions = [positions]
            if p.get("position") and p.get("position") not in positions:
                positions = list(positions) + [p.get("position")]
            if not positions and pid.isalpha() and len(pid) <= 4:
                positions = ["DEF"]
            position_map[pid] = {str(x) for x in positions if x}

    return weekly_points, position_map, score_key, good_weeks


def average_points(weekly_points: dict[int, dict[str, float]], weeks: list[int]) -> dict[str, float]:
    pids = set()
    for week in weeks:
        pids.update(weekly_points.get(week, {}).keys())

    if not weeks:
        return {}

    return {
        pid: sum(float(weekly_points.get(w, {}).get(pid, 0.0) or 0.0) for w in weeks) / len(weeks)
        for pid in pids
    }


def combo_list(players: list[str], max_assets: int) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for n in range(1, max_assets + 1):
        result.extend(itertools.combinations(players, n))
    return result


def _apply_trade(
    roster: list[str],
    outgoing: tuple[str, ...],
    incoming: tuple[str, ...],
) -> list[str]:
    out = set(outgoing)
    result = [p for p in roster if p not in out]
    result.extend(p for p in incoming if p not in result)
    return result


@dataclass
class TradeResult:
    give: tuple[str, ...]
    receive: tuple[str, ...]
    my_gain: float
    their_gain: float
    my_after: float
    their_after: float
    my_before: float
    their_before: float
    raw_sent_ppg: float
    raw_received_ppg: float

    @property
    def mutual_gain(self) -> float:
        return min(self.my_gain, self.their_gain)

    @property
    def fairness_gap(self) -> float:
        return abs(self.my_gain - self.their_gain)


def find_trades(
    my_roster: list[str],
    their_roster: list[str],
    slots: list[str],
    avg_ppg: dict[str, float],
    weekly_points: dict[int, dict[str, float]],
    projection_weeks: list[int],
    position_map: dict[str, set[str]],
    max_assets_each_side: int = 2,
    candidate_pool_size: int = 14,
    raw_value_ratio_low: float = 0.55,
    raw_value_ratio_high: float = 1.80,
    exact_rescore_limit: int = 250,
) -> list[TradeResult]:
    """
    Two-stage search:
      1) Screen every combination using each player's average projected weekly points.
      2) Re-score the best candidates week-by-week using optimized starting lineups.

    This lets positional need create positive-sum trades while keeping 2-for-2
    searches fast.
    """
    def top_pool(roster: list[str]) -> list[str]:
        # Keep highest-projected assets. If roster is smaller than the limit, keep all.
        return sorted(
            list(dict.fromkeys(roster)),
            key=lambda p: avg_ppg.get(p, 0.0),
            reverse=True,
        )[:candidate_pool_size]

    my_pool = top_pool(my_roster)
    their_pool = top_pool(their_roster)

    my_before_static, _ = optimal_lineup(my_roster, slots, avg_ppg, position_map)
    their_before_static, _ = optimal_lineup(their_roster, slots, avg_ppg, position_map)

    my_combos = combo_list(my_pool, max_assets_each_side)
    their_combos = combo_list(their_pool, max_assets_each_side)

    screened = []

    for give in my_combos:
        sent = sum(avg_ppg.get(p, 0.0) for p in give)
        for receive in their_combos:
            received = sum(avg_ppg.get(p, 0.0) for p in receive)

            if sent <= 0 and received <= 0:
                continue
            ratio = received / sent if sent > 0 else float("inf")
            if not (raw_value_ratio_low <= ratio <= raw_value_ratio_high):
                continue

            new_mine = _apply_trade(my_roster, give, receive)
            new_theirs = _apply_trade(their_roster, receive, give)

            my_after_static, _ = optimal_lineup(new_mine, slots, avg_ppg, position_map)
            their_after_static, _ = optimal_lineup(new_theirs, slots, avg_ppg, position_map)

            my_delta = my_after_static - my_before_static
            their_delta = their_after_static - their_before_static

            # Keep mutual wins plus near-wins for exact rescoring.
            screen_score = min(my_delta, their_delta) - 0.15 * abs(my_delta - their_delta)
            screened.append(
                (
                    screen_score,
                    give,
                    receive,
                    sent,
                    received,
                    new_mine,
                    new_theirs,
                )
            )

    screened.sort(key=lambda x: x[0], reverse=True)
    screened = screened[:exact_rescore_limit]

    if projection_weeks:
        my_week_baseline = {}
        their_week_baseline = {}
        for week in projection_weeks:
            pts = weekly_points.get(week, {})
            my_week_baseline[week] = optimal_lineup(my_roster, slots, pts, position_map)[0]
            their_week_baseline[week] = optimal_lineup(their_roster, slots, pts, position_map)[0]

        my_before = sum(my_week_baseline.values()) / len(projection_weeks)
        their_before = sum(their_week_baseline.values()) / len(projection_weeks)
    else:
        my_before = my_before_static
        their_before = their_before_static

    results: list[TradeResult] = []
    for _, give, receive, sent, received, new_mine, new_theirs in screened:
        if projection_weeks:
            my_after_scores = []
            their_after_scores = []
            for week in projection_weeks:
                pts = weekly_points.get(week, {})
                my_after_scores.append(optimal_lineup(new_mine, slots, pts, position_map)[0])
                their_after_scores.append(optimal_lineup(new_theirs, slots, pts, position_map)[0])
            my_after = sum(my_after_scores) / len(my_after_scores)
            their_after = sum(their_after_scores) / len(their_after_scores)
        else:
            my_after = optimal_lineup(new_mine, slots, avg_ppg, position_map)[0]
            their_after = optimal_lineup(new_theirs, slots, avg_ppg, position_map)[0]

        results.append(
            TradeResult(
                give=give,
                receive=receive,
                my_gain=my_after - my_before,
                their_gain=their_after - their_before,
                my_after=my_after,
                their_after=their_after,
                my_before=my_before,
                their_before=their_before,
                raw_sent_ppg=sent,
                raw_received_ppg=received,
            )
        )

    results.sort(
        key=lambda r: (r.mutual_gain, -r.fairness_gap, r.my_gain + r.their_gain),
        reverse=True,
    )
    return results


def trades_to_dataframe(results: list[TradeResult], players: dict) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "You give": ", ".join(player_display_name(p, players) for p in r.give),
                "You receive": ", ".join(player_display_name(p, players) for p in r.receive),
                "Your lineup Δ / wk": round(r.my_gain, 2),
                "Their lineup Δ / wk": round(r.their_gain, 2),
                "Mutual gain": round(r.mutual_gain, 2),
                "Fairness gap": round(r.fairness_gap, 2),
                "Raw PPG sent": round(r.raw_sent_ppg, 2),
                "Raw PPG received": round(r.raw_received_ppg, 2),
                "Your projected lineup": round(r.my_after, 2),
                "Their projected lineup": round(r.their_after, 2),
            }
        )
    return pd.DataFrame(rows)


def collect_full_league_snapshot(
    client: SleeperClient,
    username_or_id: str,
    league_id: str,
    season: int,
    weeks: Iterable[int] = range(1, 19),
    include_player_catalog: bool = False,
) -> dict:
    """
    Collect the league-related public data exposed by Sleeper:
      league, users, rosters, matchups, transactions, playoff brackets,
      traded picks, drafts and draft picks.

    The full NFL player catalog can optionally be included, but is large.
    """
    user = client.get_user(username_or_id)
    league = client.get_league(league_id)
    users = client.get_league_users(league_id)
    rosters = client.get_rosters(league_id)

    matchups = {}
    transactions = {}
    for week in weeks:
        try:
            matchups[str(week)] = client.get_matchups(league_id, int(week))
        except SleeperAPIError:
            matchups[str(week)] = []
        try:
            transactions[str(week)] = client.get_transactions(league_id, int(week))
        except SleeperAPIError:
            transactions[str(week)] = []

    try:
        winners = client.get_winners_bracket(league_id)
    except SleeperAPIError:
        winners = []

    try:
        losers = client.get_losers_bracket(league_id)
    except SleeperAPIError:
        losers = []

    try:
        traded_picks = client.get_traded_picks(league_id)
    except SleeperAPIError:
        traded_picks = []

    try:
        drafts = client.get_drafts(league_id)
    except SleeperAPIError:
        drafts = []

    draft_details = {}
    for draft in drafts:
        draft_id = str(draft.get("draft_id"))
        if not draft_id:
            continue
        try:
            detail = client.get_draft(draft_id)
        except SleeperAPIError:
            detail = draft
        try:
            picks = client.get_draft_picks(draft_id)
        except SleeperAPIError:
            picks = []
        try:
            traded = client.get_draft_traded_picks(draft_id)
        except SleeperAPIError:
            traded = []
        draft_details[draft_id] = {
            "draft": detail,
            "picks": picks,
            "traded_picks": traded,
        }

    snapshot = {
        "collected_at_unix": int(time.time()),
        "user": user,
        "league": league,
        "league_users": users,
        "rosters": rosters,
        "matchups_by_week": matchups,
        "transactions_by_week": transactions,
        "winners_bracket": winners,
        "losers_bracket": losers,
        "traded_picks": traded_picks,
        "drafts": draft_details,
    }

    if include_player_catalog:
        snapshot["nfl_players"] = client.get_players()

    return snapshot


def save_snapshot(snapshot: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return path
