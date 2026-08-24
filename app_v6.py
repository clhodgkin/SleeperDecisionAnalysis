
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from sleeper_trade import (
    SleeperAPIError,
    SleeperClient,
    average_points,
    build_team_maps,
    collect_full_league_snapshot,
    find_trades,
    normalize_projection_payload,
    optimal_lineup,
    player_display_name,
    projection_position,
    scoring_key_from_league,
    starter_slots,
    trades_to_dataframe,
)


st.set_page_config(page_title="Sleeper Trade Finder", page_icon="🏈", layout="wide")
st.title("🏈 Sleeper Mutual-Gain Trade Finder")
st.caption(
    "Find trades that can improve both teams' projected optimal starting lineup, "
    "using projected average points per week."
)

st.caption("**Build v6 — blank username + balanced trade sizes only**")

DEFAULT_USERNAME = ""
PREFERRED_LEAGUE_ID = "1393026126246346752"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_user_and_leagues(username: str, season: int):
    client = SleeperClient()
    user = client.get_user(username)
    leagues = client.get_leagues(str(user["user_id"]), season)
    return user, leagues


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_league_core(league_id: str):
    client = SleeperClient()
    league = client.get_league(league_id)
    users = client.get_league_users(league_id)
    rosters = client.get_rosters(league_id)
    players = client.get_players()
    return league, users, rosters, players


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_projection_bundle(league_id: str, season: int):
    """
    Fetch weekly Sleeper projections once, then build two scoring views:

    sleeper_points:
        Sleeper's generic pts_ppr / pts_half_ppr / pts_std field.

    league_points:
        Raw projected stat line re-scored with this league's scoring_settings.

    projected_game:
        Tracks whether the player is projected to play that week. This lets
        season PPG exclude bye weeks instead of treating the bye as a 0-point game.
    """
    client = SleeperClient()
    league = client.get_league(league_id)
    players = client.get_players()

    score_key = scoring_key_from_league(league)
    scoring_settings = league.get("scoring_settings") or {}

    league_points = {}
    sleeper_points = {}
    projected_game = {}
    position_map = {}
    available_stat_keys = set()
    good_weeks = []

    for week in range(1, 19):
        payload = client.get_week_projections(season, week)
        normalized = normalize_projection_payload(payload)
        if not normalized:
            continue

        good_weeks.append(week)
        league_points[week] = {}
        sleeper_points[week] = {}
        projected_game[week] = {}

        for pid, record in normalized.items():
            stats = record.get("stats") or {}
            raw = record.get("raw") or {}
            available_stat_keys.update(stats.keys())

            # Sleeper's generic format projection. This is useful as a direct
            # comparison against the underlying projection feed.
            generic_value = stats.get(score_key, raw.get(score_key, 0.0))
            try:
                generic_value = float(generic_value or 0.0)
            except (TypeError, ValueError):
                generic_value = 0.0
            sleeper_points[week][pid] = generic_value

            # Re-score the raw projected stats using the league's actual settings.
            # This is a straight dot product over the stat keys Sleeper exposes.
            custom_total = 0.0
            matched_rules = 0
            for stat_name, multiplier in scoring_settings.items():
                if stat_name not in stats:
                    continue
                try:
                    custom_total += (
                        float(stats.get(stat_name, 0.0) or 0.0)
                        * float(multiplier or 0.0)
                    )
                    matched_rules += 1
                except (TypeError, ValueError):
                    pass

            # If for some unusual payload no league scoring fields matched at all,
            # fall back to Sleeper's generic value instead of showing a false zero.
            league_points[week][pid] = (
                float(custom_total) if matched_rules else generic_value
            )

            # Weekly projection payloads commonly include gp. Use it when present.
            # If it is absent, presence of a meaningful projection is the fallback.
            gp = stats.get("gp", raw.get("gp"))
            if gp is not None:
                try:
                    is_game = float(gp) > 0
                except (TypeError, ValueError):
                    is_game = True
            else:
                is_game = (
                    abs(generic_value) > 1e-9
                    or abs(float(league_points[week][pid])) > 1e-9
                )
            projected_game[week][pid] = is_game

            position_map[pid] = projection_position(record, players, pid)

    # Ensure every rostered/master-catalog player can still be assigned a position
    # even if that player is absent from the projection feed.
    for pid, p in players.items():
        if pid in position_map:
            continue

        positions = p.get("fantasy_positions") or []
        if isinstance(positions, str):
            positions = [positions]
        if p.get("position") and p.get("position") not in positions:
            positions = list(positions) + [p.get("position")]
        if not positions and str(pid).isalpha() and len(str(pid)) <= 4:
            positions = ["DEF"]
        position_map[str(pid)] = {str(x) for x in positions if x}

    nonzero_rules = {
        key
        for key, value in scoring_settings.items()
        if value not in (None, 0, 0.0)
    }
    unmatched_rules = sorted(nonzero_rules - available_stat_keys)

    return (
        league_points,
        sleeper_points,
        projected_game,
        position_map,
        score_key,
        good_weeks,
        unmatched_rules,
    )


def per_game_average(
    weekly_points: dict[int, dict[str, float]],
    projected_game: dict[int, dict[str, bool]],
    weeks: list[int],
) -> dict[str, float]:
    """
    Average projected points per projected NFL game, not per calendar week.

    This is the important bye-week fix: a player's bye is not counted as a
    zero-point game when displaying season PPG or screening raw trade value.
    """
    player_ids = set()
    for week in weeks:
        player_ids.update(weekly_points.get(week, {}).keys())

    result = {}
    for pid in player_ids:
        values = []
        for week in weeks:
            if pid not in weekly_points.get(week, {}):
                continue
            if not projected_game.get(week, {}).get(pid, True):
                continue
            values.append(float(weekly_points[week].get(pid, 0.0) or 0.0))

        # Fallback for unusual payloads where gp is not populated correctly.
        if not values:
            values = [
                float(weekly_points.get(week, {}).get(pid, 0.0) or 0.0)
                for week in weeks
                if pid in weekly_points.get(week, {})
            ]

        result[pid] = sum(values) / len(values) if values else 0.0

    return result


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_nfl_state():
    return SleeperClient().get_nfl_state()


def keep_equal_size_trades(results, trade_size_mode: str):
    """
    Restrict recommendations to balanced roster-count trades.

    Supported:
      - 1-for-1
      - 2-for-2
      - both

    Uneven 1-for-2 / 2-for-1 trades are intentionally excluded because they
    require modeling a roster cut/open roster spot to evaluate correctly.
    """
    filtered = []

    for result in results:
        give_count = len(result.give)
        receive_count = len(result.receive)

        if give_count != receive_count:
            continue

        if trade_size_mode == "1-for-1 only" and give_count != 1:
            continue

        if trade_size_mode == "2-for-2 only" and give_count != 2:
            continue

        if trade_size_mode == "1-for-1 and 2-for-2" and give_count not in (1, 2):
            continue

        filtered.append(result)

    return filtered


@st.cache_data(ttl=5 * 60, show_spinner=False)
def fetch_week_matchups(league_id: str, week: int):
    """Fetch the league's saved matchup/lineup data for one NFL week."""
    return SleeperClient().get_matchups(league_id, int(week))


def matchup_projection_rows(
    matchup_row: dict,
    fallback_team: dict,
    lineup_slots: list[str],
    week: int,
    weekly_points: dict[int, dict[str, float]],
    sleeper_weekly_points: dict[int, dict[str, float]],
    players: dict,
    position_map: dict[str, set[str]],
) -> list[dict]:
    """
    Build a player-by-player projection table using Sleeper's saved starters
    for the selected matchup week. If the matchup row has no starters yet,
    fall back to the roster's current saved starters.
    """
    starter_ids = [
        str(pid)
        for pid in (
            (matchup_row or {}).get("starters")
            or fallback_team.get("starters")
            or []
        )
        if pid not in (None, "0")
    ]

    rows = []
    for idx, pid in enumerate(starter_ids):
        slot = lineup_slots[idx] if idx < len(lineup_slots) else "START"
        league_proj = float(weekly_points.get(week, {}).get(pid, 0.0) or 0.0)
        sleeper_base = float(
            sleeper_weekly_points.get(week, {}).get(pid, 0.0) or 0.0
        )

        rows.append(
            {
                "Slot": slot,
                "Player": player_display_name(pid, players),
                "Pos": "/".join(sorted(position_map.get(pid, set()))),
                "League-rule projection": round(league_proj, 2),
                "Sleeper base": round(sleeper_base, 2),
                "Rule adjustment": round(league_proj - sleeper_base, 2),
            }
        )

    return rows


with st.sidebar:
    st.header("League")
    username = st.text_input("Sleeper username", value=DEFAULT_USERNAME, placeholder="your_username")
    season = st.number_input("Season", min_value=2018, max_value=2100, value=datetime.now().year, step=1)

    st.caption(
        "Trade scoring uses this league's scoring_settings applied to Sleeper's "
        "raw weekly projection stats."
    )

if not username:
    st.info("Enter your Sleeper username in the sidebar to load your leagues.")
    st.stop()

try:
    user, leagues = fetch_user_and_leagues(username.strip(), int(season))
except Exception as exc:
    st.error(f"Could not load Sleeper user/leagues: {exc}")
    st.stop()

if not leagues:
    st.warning(f"No NFL leagues found for {username} in {season}.")
    st.stop()

league_labels = {
    f"{lg.get('name', 'Unnamed league')} — {lg.get('league_id')}": str(lg["league_id"])
    for lg in leagues
}
league_options = list(league_labels.keys())
preferred_index = 0
for i, label in enumerate(league_options):
    if league_labels[label] == PREFERRED_LEAGUE_ID:
        preferred_index = i
        break

selected_label = st.sidebar.selectbox(
    "League",
    league_options,
    index=preferred_index,
)
league_id = league_labels[selected_label]

try:
    league, users, rosters, players = fetch_league_core(league_id)
except Exception as exc:
    st.error(f"Could not load league data: {exc}")
    st.stop()

teams, owner_to_roster = build_team_maps(users, rosters)
my_roster_id = owner_to_roster.get(str(user["user_id"]))
if my_roster_id is None:
    st.error("I found the league, but could not match your Sleeper user to a roster.")
    st.stop()

my_team = teams[my_roster_id]

st.subheader(league.get("name", "Sleeper League"))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Teams", league.get("total_rosters", len(rosters)))
c2.metric("Your team", my_team["team_name"])
c3.metric("Season", league.get("season", season))
c4.metric("Starter slots", len(starter_slots(league)))

with st.expander("League settings / roster slots"):
    st.write("Roster positions:", league.get("roster_positions"))
    st.json(league.get("scoring_settings") or {})

with st.spinner("Loading and re-scoring weekly projections..."):
    try:
        (
            weekly_points,
            sleeper_weekly_points,
            projected_game,
            position_map,
            score_key,
            projection_weeks,
            unmatched_scoring_rules,
        ) = fetch_projection_bundle(league_id, int(season))
    except Exception as exc:
        st.error(
            "Could not load Sleeper projections. The projection feed is unofficial "
            f"and may have changed. Details: {exc}"
        )
        st.stop()

if not projection_weeks:
    st.error("No weekly projection data was returned.")
    st.stop()

# Correct season PPG: average projected GAMES, not all 18 calendar weeks.
avg_ppg = per_game_average(
    weekly_points,
    projected_game,
    projection_weeks,
)
sleeper_avg_ppg = per_game_average(
    sleeper_weekly_points,
    projected_game,
    projection_weeks,
)

slots = starter_slots(league)

try:
    nfl_state = fetch_nfl_state()
    default_compare_week = int(nfl_state.get("display_week") or 1)
except Exception:
    default_compare_week = 1

if default_compare_week not in projection_weeks:
    default_compare_week = projection_weeks[0]

if (
    "active_compare_week" not in st.session_state
    or st.session_state.active_compare_week not in projection_weeks
):
    st.session_state.active_compare_week = default_compare_week

with st.form("week_comparison_controls"):
    chosen_week = st.selectbox(
        "Week to inspect",
        projection_weeks,
        index=projection_weeks.index(st.session_state.active_compare_week),
        help="Choose a week, then click Load selected week.",
    )
    week_col1, week_col2 = st.columns(2)
    with week_col1:
        load_week = st.form_submit_button(
            "Load selected week",
            type="primary",
            use_container_width=True,
        )
    with week_col2:
        refresh_week = st.form_submit_button(
            "Refresh from Sleeper",
            use_container_width=True,
            help="Clears cached league, projection, and matchup data and fetches it again.",
        )

if load_week:
    st.session_state.active_compare_week = int(chosen_week)

if refresh_week:
    st.session_state.active_compare_week = int(chosen_week)
    fetch_user_and_leagues.clear()
    fetch_league_core.clear()
    fetch_projection_bundle.clear()
    fetch_nfl_state.clear()
    fetch_week_matchups.clear()
    st.rerun()

compare_week = int(st.session_state.active_compare_week)
st.info(f"Showing **Week {compare_week}** in the matchup projection check below.")

st.caption(
    f"Projection weeks loaded: {min(projection_weeks)}–{max(projection_weeks)} "
    f"({len(projection_weeks)} weeks). Sleeper base field: `{score_key}`. "
    "Trade calculations use league-rule rescoring."
)

if unmatched_scoring_rules:
    st.warning(
        "Projection scoring audit: Sleeper's raw projection payload does not expose "
        "every non-zero scoring category configured in this league. Those categories "
        "cannot be projected exactly from the available feed. Missing keys: "
        + ", ".join(unmatched_scoring_rules)
    )
else:
    st.success(
        "Projection scoring audit: every non-zero league scoring key appears in "
        "Sleeper's raw projection data. The app is re-scoring projections from the "
        "league settings rather than relying only on generic PPR."
    )

with st.expander("Why these projections may differ from the old version"):
    st.write(
        "The old app divided player projections across all 18 NFL weeks, which "
        "effectively treated a bye week like a 0-point game. The Season PPG shown "
        "now averages only weeks where Sleeper marks the player as projected to play."
    )
    st.write(
        "The trade engine also now uses the league's scoring_settings against the "
        "raw projected stat categories. The 'Sleeper base' comparison columns below "
        "show the generic PPR/half-PPR/standard number from the projection feed."
    )

# Roster overview
# Show both total roster projection and the points each team can actually place
# into a legal starting lineup. This makes excess depth visible instead of
# making deep benches look artificially stronger week-to-week.
roster_rows = []
for rid, team in teams.items():
    projected_roster_total = sum(avg_ppg.get(pid, 0.0) for pid in team["players"])

    weekly_optimal_scores = []
    for week in projection_weeks:
        week_points = weekly_points.get(week, {})
        weekly_score, _ = optimal_lineup(
            team["players"],
            slots,
            week_points,
            position_map,
        )
        weekly_optimal_scores.append(weekly_score)

    optimal_starting_ppg = (
        sum(weekly_optimal_scores) / len(weekly_optimal_scores)
        if weekly_optimal_scores
        else 0.0
    )
    bench_value = max(0.0, projected_roster_total - optimal_starting_ppg)

    roster_rows.append(
        {
            "Roster": rid,
            "Team": team["team_name"],
            "Manager": team.get("display_name") or team.get("username") or "",
            "Players": len(team["players"]),
            "Optimal Lineup / Fantasy Week": round(optimal_starting_ppg, 2),
            "Total Player PPG": round(projected_roster_total, 2),
            "Unused Player PPG": round(bench_value, 2),
        }
    )

roster_overview_df = pd.DataFrame(roster_rows).sort_values(
    "Optimal Lineup / Fantasy Week",
    ascending=False,
)

st.markdown("### Season roster overview")
st.caption("This table is season-wide, so it will not change when you switch the matchup week above.")

st.dataframe(
    roster_overview_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Optimal Lineup / Fantasy Week": st.column_config.NumberColumn(
            "Optimal Lineup / Fantasy Week",
            help=(
                "Average projected points from the best legal starting lineup "
                "for each projection week."
            ),
            format="%.2f",
        ),
        "Total Player PPG": st.column_config.NumberColumn(
            "Total Player PPG",
            help=(
                "Sum of every rostered player's projected points per NFL game, "
                "including bench players. Player byes are excluded from each "
                "player's PPG denominator."
            ),
            format="%.2f",
        ),
        "Unused Player PPG": st.column_config.NumberColumn(
            "Unused Player PPG",
            help=(
                "Difference between total player PPG and usable optimal-lineup projection. "
                "A high value can indicate excess/tradeable depth."
            ),
            format="%.2f",
        ),
    },
)

st.markdown(f"### Week {compare_week} matchup projection check")
st.caption(
    "This is the cleanest accuracy test. Choose the same week in Sleeper's Matchup "
    "tab and compare the player/team projections shown there with the league-rule "
    "projection below."
)

try:
    selected_matchups = fetch_week_matchups(league_id, int(compare_week))
except Exception as exc:
    selected_matchups = []
    st.warning(f"Could not load Sleeper matchup data for Week {compare_week}: {exc}")

my_matchup_row = next(
    (
        row
        for row in selected_matchups
        if int(row.get("roster_id", -1)) == int(my_roster_id)
    ),
    None,
)

opponent_matchup_row = None
matchup_opponent_team = None

if my_matchup_row is not None and my_matchup_row.get("matchup_id") is not None:
    matchup_id = my_matchup_row.get("matchup_id")
    opponent_matchup_row = next(
        (
            row
            for row in selected_matchups
            if row.get("matchup_id") == matchup_id
            and int(row.get("roster_id", -1)) != int(my_roster_id)
        ),
        None,
    )

    if opponent_matchup_row is not None:
        opponent_matchup_roster_id = int(opponent_matchup_row["roster_id"])
        matchup_opponent_team = teams.get(opponent_matchup_roster_id)

my_week_rows = matchup_projection_rows(
    matchup_row=my_matchup_row or {},
    fallback_team=my_team,
    lineup_slots=slots,
    week=int(compare_week),
    weekly_points=weekly_points,
    sleeper_weekly_points=sleeper_weekly_points,
    players=players,
    position_map=position_map,
)

my_league_total = sum(
    float(row["League-rule projection"]) for row in my_week_rows
)
my_base_total = sum(float(row["Sleeper base"]) for row in my_week_rows)

if matchup_opponent_team is not None:
    opp_week_rows = matchup_projection_rows(
        matchup_row=opponent_matchup_row or {},
        fallback_team=matchup_opponent_team,
        lineup_slots=slots,
        week=int(compare_week),
        weekly_points=weekly_points,
        sleeper_weekly_points=sleeper_weekly_points,
        players=players,
        position_map=position_map,
    )
    opp_league_total = sum(
        float(row["League-rule projection"]) for row in opp_week_rows
    )
    opp_base_total = sum(float(row["Sleeper base"]) for row in opp_week_rows)
else:
    opp_week_rows = []
    opp_league_total = 0.0
    opp_base_total = 0.0

if my_week_rows:
    matchup_name = (
        matchup_opponent_team["team_name"]
        if matchup_opponent_team is not None
        else "Opponent not yet available"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"Your Week {compare_week} projection",
        f"{my_league_total:.2f}",
        help="Sum of your saved starters after applying this league's scoring rules.",
    )
    if matchup_opponent_team is not None:
        m2.metric(
            f"{matchup_name} projection",
            f"{opp_league_total:.2f}",
            help="Opponent's saved starters scored with the same league rules.",
        )
        m3.metric(
            "Projected margin",
            f"{my_league_total - opp_league_total:+.2f}",
        )
    else:
        m2.metric("Sleeper generic total", f"{my_base_total:.2f}")
        m3.metric(
            "League-rule adjustment",
            f"{my_league_total - my_base_total:+.2f}",
        )

    matchup_left, matchup_right = st.columns(2)

    with matchup_left:
        st.markdown(f"#### {my_team['team_name']}")
        my_validation_df = pd.DataFrame(my_week_rows)
        my_validation_df["Sleeper app displayed"] = None

        edited_my = st.data_editor(
            my_validation_df,
            use_container_width=True,
            hide_index=True,
            disabled=[
                "Slot",
                "Player",
                "Pos",
                "League-rule projection",
                "Sleeper base",
                "Rule adjustment",
            ],
            column_config={
                "League-rule projection": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Sleeper base": st.column_config.NumberColumn(format="%.2f"),
                "Rule adjustment": st.column_config.NumberColumn(format="%+.2f"),
                "Sleeper app displayed": st.column_config.NumberColumn(
                    "Sleeper app displayed",
                    help=(
                        "Optional: type the projection shown in the Sleeper app "
                        "for this player to calculate an exact difference."
                    ),
                    format="%.2f",
                ),
            },
            key=f"my_projection_audit_{compare_week}",
        )

        entered_my = edited_my.dropna(subset=["Sleeper app displayed"]).copy()
        if not entered_my.empty:
            entered_my["App difference"] = (
                entered_my["League-rule projection"]
                - entered_my["Sleeper app displayed"]
            ).round(2)
            max_abs_diff = entered_my["App difference"].abs().max()
            mean_abs_diff = entered_my["App difference"].abs().mean()

            st.write(
                f"Entered-player mean absolute difference: **{mean_abs_diff:.2f}** "
                f"points; largest difference: **{max_abs_diff:.2f}**."
            )
            st.dataframe(
                entered_my[
                    [
                        "Player",
                        "League-rule projection",
                        "Sleeper app displayed",
                        "App difference",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with matchup_right:
        if matchup_opponent_team is not None and opp_week_rows:
            st.markdown(f"#### {matchup_opponent_team['team_name']}")
            st.dataframe(
                pd.DataFrame(opp_week_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "League-rule projection": st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                    "Sleeper base": st.column_config.NumberColumn(format="%.2f"),
                    "Rule adjustment": st.column_config.NumberColumn(
                        format="%+.2f"
                    ),
                },
            )
        else:
            st.markdown("#### Matchup opponent")
            st.info(
                "Sleeper did not return a paired opponent for this week yet. "
                "Your saved lineup can still be checked player-by-player."
            )

    with st.expander("How to validate against Sleeper"):
        st.markdown(
            f"""
1. Open Sleeper and go to this league's **Matchup** screen.
2. Select **Week {compare_week}**.
3. Compare each starter's projected points with **League-rule projection** above.
4. Optionally type Sleeper's displayed number into **Sleeper app displayed**.
5. If the differences are consistently near 0.00, the scorer is reproducing
   Sleeper accurately. If specific positions or players differ, the pattern
   tells us which scoring category still needs adjustment.
"""
        )
else:
    st.info(
        f"No saved starters were returned for Week {compare_week}. "
        "Try another week or set a lineup in Sleeper first."
    )

opponents = {
    team["team_name"]: rid
    for rid, team in teams.items()
    if rid != my_roster_id
}

st.markdown("### Trade explorer")
search_scope = st.radio(
    "Search scope",
    ["One trade partner", "Entire league"],
    horizontal=True,
    help=(
        "One trade partner compares your roster against one manager. Entire league "
        "runs the same mutual-gain search against every other roster."
    ),
)

opp_team = None
if search_scope == "One trade partner":
    opponent_name = st.selectbox("Trade partner", list(opponents.keys()))
    opp_id = opponents[opponent_name]
    opp_team = teams[opp_id]

left, right = st.columns(2)
with left:
    st.markdown(f"### Your roster — {my_team['team_name']}")
    my_rows = [
        {
            "Player": player_display_name(pid, players),
            "Pos": "/".join(sorted(position_map.get(pid, set()))),
            "Season PPG (league rules)": round(avg_ppg.get(pid, 0.0), 2),
            f"Wk {compare_week} (league rules)": round(
                weekly_points.get(compare_week, {}).get(pid, 0.0), 2
            ),
            f"Wk {compare_week} (Sleeper base)": round(
                sleeper_weekly_points.get(compare_week, {}).get(pid, 0.0), 2
            ),
        }
        for pid in sorted(
            my_team["players"],
            key=lambda p: avg_ppg.get(p, 0),
            reverse=True,
        )
    ]
    st.dataframe(
        pd.DataFrame(my_rows),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Season PPG is an average across projected games; use the Weekly matchup "
        "projection check above for an apples-to-apples Sleeper comparison."
    )

with right:
    if search_scope == "One trade partner":
        st.markdown(f"### Their roster — {opp_team['team_name']}")
        opp_rows = [
            {
                "Player": player_display_name(pid, players),
                "Pos": "/".join(sorted(position_map.get(pid, set()))),
                "Season PPG (league rules)": round(avg_ppg.get(pid, 0.0), 2),
                f"Wk {compare_week} (league rules)": round(
                    weekly_points.get(compare_week, {}).get(pid, 0.0), 2
                ),
                f"Wk {compare_week} (Sleeper base)": round(
                    sleeper_weekly_points.get(compare_week, {}).get(pid, 0.0), 2
                ),
            }
            for pid in sorted(
                opp_team["players"],
                key=lambda p: avg_ppg.get(p, 0),
                reverse=True,
            )
        ]
        st.dataframe(
            pd.DataFrame(opp_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown("### League-wide mode")
        st.write(
            "The search will compare your roster independently against all "
            f"{len(opponents)} other managers and combine the best results."
        )
        st.dataframe(
            roster_overview_df[
                ["Team", "Manager", "Optimal Lineup / Fantasy Week", "Unused Player PPG"]
            ],
            use_container_width=True,
            hide_index=True,
        )

st.markdown("### Trade search settings")
st.caption(
    "Deployment mode: only 1-for-1 and 2-for-2 trades are evaluated as valid "
    "recommendations. Uneven trades are hidden."
)
a, b, c = st.columns(3)
with a:
    trade_size_mode = st.selectbox(
        "Trade sizes",
        [
            "1-for-1 and 2-for-2",
            "1-for-1 only",
            "2-for-2 only",
        ],
        index=0,
        help=(
            "Uneven trades are intentionally excluded until roster cuts/open "
            "roster spots are modeled explicitly."
        ),
    )
    max_assets = 1 if trade_size_mode == "1-for-1 only" else 2
with b:
    max_roster_size = max(len(t["players"]) for t in teams.values())
    candidate_pool = st.slider(
        "Players considered per roster",
        min_value=6,
        max_value=max(max_roster_size, 6),
        value=min(14, max_roster_size),
    )
with c:
    exact_limit = st.slider(
        "Final candidates to exact-score",
        50,
        500,
        250,
        50,
        help=(
            "Applied per opponent. League-wide searches can take longer at "
            "higher values."
        ),
    )

ratio_low, ratio_high = st.slider(
    "Raw projected-value screening ratio (received / sent)",
    min_value=0.25,
    max_value=3.00,
    value=(0.55, 1.80),
    step=0.05,
)

button_text = (
    "Find trades with selected manager"
    if search_scope == "One trade partner"
    else "Search entire league for trades"
)

if st.button(button_text, type="primary"):
    if search_scope == "One trade partner":
        with st.spinner("Evaluating trade combinations..."):
            results = find_trades(
                my_roster=my_team["players"],
                their_roster=opp_team["players"],
                slots=slots,
                avg_ppg=avg_ppg,
                weekly_points=weekly_points,
                projection_weeks=projection_weeks,
                position_map=position_map,
                max_assets_each_side=int(max_assets),
                candidate_pool_size=int(candidate_pool),
                raw_value_ratio_low=float(ratio_low),
                raw_value_ratio_high=float(ratio_high),
                exact_rescore_limit=int(exact_limit),
            )

        results = keep_equal_size_trades(results, trade_size_mode)
        results = keep_equal_size_trades(results, trade_size_mode)
        mutual = [r for r in results if r.my_gain > 0 and r.their_gain > 0]
        display_results = mutual if mutual else results[:50]

        if mutual:
            st.success(
                f"Found {len(mutual)} trade(s) that improve both projected "
                "starting lineups."
            )
        else:
            st.warning(
                "No strictly positive-for-both trades survived the current "
                "filters. Showing the closest options instead."
            )

        df = trades_to_dataframe(display_results[:100], players)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Your lineup Δ / wk": st.column_config.NumberColumn(format="%+.2f"),
                "Their lineup Δ / wk": st.column_config.NumberColumn(format="%+.2f"),
                "Mutual gain": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

    else:
        league_rows = []
        total_opponents = len(opponents)
        progress = st.progress(0, text="Starting league-wide search...")

        for i, (team_name, rid) in enumerate(opponents.items(), start=1):
            other_team = teams[rid]
            progress.progress(
                int((i - 1) / max(total_opponents, 1) * 100),
                text=f"Searching trades with {team_name} ({i}/{total_opponents})...",
            )

            results = find_trades(
                my_roster=my_team["players"],
                their_roster=other_team["players"],
                slots=slots,
                avg_ppg=avg_ppg,
                weekly_points=weekly_points,
                projection_weeks=projection_weeks,
                position_map=position_map,
                max_assets_each_side=int(max_assets),
                candidate_pool_size=int(candidate_pool),
                raw_value_ratio_low=float(ratio_low),
                raw_value_ratio_high=float(ratio_high),
                exact_rescore_limit=int(exact_limit),
            )

            mutual = [r for r in results if r.my_gain > 0 and r.their_gain > 0]
            for result in mutual[:50]:
                row = {
                    "Trade partner": team_name,
                    "Manager": (
                        other_team.get("display_name")
                        or other_team.get("username")
                        or ""
                    ),
                    "You give": ", ".join(
                        player_display_name(p, players) for p in result.give
                    ),
                    "You receive": ", ".join(
                        player_display_name(p, players) for p in result.receive
                    ),
                    "Your lineup Δ / wk": round(result.my_gain, 2),
                    "Their lineup Δ / wk": round(result.their_gain, 2),
                    "Mutual gain": round(result.mutual_gain, 2),
                    "Fairness gap": round(result.fairness_gap, 2),
                    "Raw PPG sent": round(result.raw_sent_ppg, 2),
                    "Raw PPG received": round(result.raw_received_ppg, 2),
                }
                league_rows.append(row)

        progress.progress(100, text="League-wide search complete.")

        if league_rows:
            league_df = pd.DataFrame(league_rows)
            league_df = league_df.sort_values(
                ["Mutual gain", "Fairness gap", "Your lineup Δ / wk"],
                ascending=[False, True, False],
            ).reset_index(drop=True)
            league_df.insert(0, "Rank", range(1, len(league_df) + 1))

            st.success(
                f"Found {len(league_df)} mutually positive trade combinations "
                f"across {total_opponents} possible trade partners."
            )
            st.dataframe(
                league_df.head(200),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Your lineup Δ / wk": st.column_config.NumberColumn(
                        format="%+.2f"
                    ),
                    "Their lineup Δ / wk": st.column_config.NumberColumn(
                        format="%+.2f"
                    ),
                    "Mutual gain": st.column_config.NumberColumn(format="%+.2f"),
                },
            )

            partner_summary = (
                league_df.groupby(["Trade partner", "Manager"], as_index=False)
                .agg(
                    Best_Mutual_Gain=("Mutual gain", "max"),
                    Positive_Trades=("Mutual gain", "count"),
                    Best_Your_Gain=("Your lineup Δ / wk", "max"),
                )
                .sort_values(
                    ["Best_Mutual_Gain", "Positive_Trades"],
                    ascending=[False, False],
                )
            )
            partner_summary.columns = [
                "Trade partner",
                "Manager",
                "Best mutual gain",
                "Positive trade ideas",
                "Best gain for you",
            ]

            st.markdown("#### Best trade partners")
            st.dataframe(
                partner_summary,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "No trades that increased both teams' projected optimal lineup "
                "were found league-wide with the current filters."
            )

    st.caption(
        "The lineup Δ values compare each team's best legal projected starting "
        "lineup before vs. after the trade across the projection weeks. Weekly "
        "lineup scoring keeps bye weeks as real zero-availability weeks; individual "
        "Season PPG excludes byes from the denominator. This is still a projection-only "
        "model and does not include subjective upside or manager preferences."
    )

st.divider()
st.markdown("### Export league data")
st.write(
    "This exports league metadata, users, rosters, weeks 1–18 matchups and transactions, "
    "playoff brackets, traded picks, drafts, and draft picks to one JSON file."
)

if st.button("Build JSON league snapshot"):
    with st.spinner("Collecting league endpoints..."):
        client = SleeperClient()
        snapshot = collect_full_league_snapshot(
            client=client,
            username_or_id=username.strip(),
            league_id=league_id,
            season=int(season),
            weeks=range(1, 19),
            include_player_catalog=False,
        )
    payload = json.dumps(snapshot, indent=2).encode("utf-8")
    st.download_button(
        "Download league_snapshot.json",
        data=payload,
        file_name=f"sleeper_league_{league_id}_{season}.json",
        mime="application/json",
    )

st.caption(
    "Note: the full NFL player catalog is intentionally not embedded in the snapshot "
    "because Sleeper's player response is large. The app fetches it separately and caches it."
)
