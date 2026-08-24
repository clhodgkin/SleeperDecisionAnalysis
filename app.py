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
    collect_projection_data,
    find_trades,
    optimal_lineup,
    player_display_name,
    starter_slots,
    trades_to_dataframe,
)


st.set_page_config(page_title="Sleeper Trade Finder", page_icon="🏈", layout="wide")
st.title("🏈 Sleeper Mutual-Gain Trade Finder")
st.caption(
    "Find trades that can improve both teams' projected optimal starting lineup, "
    "using projected average points per week."
)

DEFAULT_USERNAME = "chodgkin"
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
def fetch_projections(league_id: str, season: int, scoring_mode: str):
    client = SleeperClient()
    league = client.get_league(league_id)
    players = client.get_players()
    weekly_points, position_map, score_key, weeks = collect_projection_data(
        client=client,
        season=season,
        weeks=range(1, 19),
        players=players,
        league=league,
        scoring_mode=scoring_mode,
    )
    return weekly_points, position_map, score_key, weeks


with st.sidebar:
    st.header("League")
    username = st.text_input("Sleeper username", value=DEFAULT_USERNAME, placeholder="your_username")
    season = st.number_input("Season", min_value=2018, max_value=2100, value=datetime.now().year, step=1)

    scoring_choice = st.selectbox(
        "Projection scoring",
        [
            "Auto from league reception scoring",
            "League settings (experimental custom scorer)",
            "PPR",
            "Half PPR",
            "Standard",
        ],
        index=0,
    )
    scoring_map = {
        "Auto from league reception scoring": "auto",
        "League settings (experimental custom scorer)": "custom",
        "PPR": "ppr",
        "Half PPR": "half_ppr",
        "Standard": "standard",
    }

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

with st.spinner("Loading weekly projections..."):
    try:
        weekly_points, position_map, score_key, projection_weeks = fetch_projections(
            league_id, int(season), scoring_map[scoring_choice]
        )
    except Exception as exc:
        st.error(
            "Could not load Sleeper projections. The projection feed is unofficial and "
            f"may have changed. Details: {exc}"
        )
        st.stop()

if not projection_weeks:
    st.error("No weekly projection data was returned.")
    st.stop()

avg_ppg = average_points(weekly_points, projection_weeks)
slots = starter_slots(league)

st.caption(
    f"Projection weeks loaded: {min(projection_weeks)}–{max(projection_weeks)} "
    f"({len(projection_weeks)} weeks). Standard projection field: `{score_key}`."
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
            "Optimal Starting PPG": round(optimal_starting_ppg, 2),
            "Total Roster PPG": round(projected_roster_total, 2),
            "Bench / Unused PPG": round(bench_value, 2),
        }
    )

roster_overview_df = pd.DataFrame(roster_rows).sort_values(
    "Optimal Starting PPG",
    ascending=False,
)

st.dataframe(
    roster_overview_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Optimal Starting PPG": st.column_config.NumberColumn(
            "Optimal Starting PPG",
            help=(
                "Average projected points from the best legal starting lineup "
                "for each projection week."
            ),
            format="%.2f",
        ),
        "Total Roster PPG": st.column_config.NumberColumn(
            "Total Roster PPG",
            help=(
                "Sum of every rostered player's average projected weekly points, "
                "including bench players."
            ),
            format="%.2f",
        ),
        "Bench / Unused PPG": st.column_config.NumberColumn(
            "Bench / Unused PPG",
            help=(
                "Projected roster points that cannot be used in the optimal "
                "starting lineup on an average week. A high number can indicate "
                "tradeable depth."
            ),
            format="%.2f",
        ),
    },
)

opponents = {team["team_name"]: rid for rid, team in teams.items() if rid != my_roster_id}
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
            "Projected PPG": round(avg_ppg.get(pid, 0.0), 2),
        }
        for pid in sorted(my_team["players"], key=lambda p: avg_ppg.get(p, 0), reverse=True)
    ]
    st.dataframe(pd.DataFrame(my_rows), use_container_width=True, hide_index=True)

with right:
    st.markdown(f"### Their roster — {opp_team['team_name']}")
    opp_rows = [
        {
            "Player": player_display_name(pid, players),
            "Pos": "/".join(sorted(position_map.get(pid, set()))),
            "Projected PPG": round(avg_ppg.get(pid, 0.0), 2),
        }
        for pid in sorted(opp_team["players"], key=lambda p: avg_ppg.get(p, 0), reverse=True)
    ]
    st.dataframe(pd.DataFrame(opp_rows), use_container_width=True, hide_index=True)

st.markdown("### Trade search")
a, b, c = st.columns(3)
with a:
    max_assets = st.selectbox("Max players from each side", [1, 2], index=1)
with b:
    candidate_pool = st.slider(
        "Players considered per roster",
        min_value=6,
        max_value=max(len(my_team["players"]), len(opp_team["players"]), 6),
        value=min(14, max(len(my_team["players"]), len(opp_team["players"]))),
    )
with c:
    exact_limit = st.slider("Final candidates to exact-score", 50, 500, 250, 50)

ratio_low, ratio_high = st.slider(
    "Raw projected-value screening ratio (received / sent)",
    min_value=0.25,
    max_value=3.00,
    value=(0.55, 1.80),
    step=0.05,
)

if st.button("Find mutually appealing trades", type="primary"):
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

    mutual = [r for r in results if r.my_gain > 0 and r.their_gain > 0]
    display_results = mutual if mutual else results[:50]

    if mutual:
        st.success(f"Found {len(mutual)} trade(s) that improve both projected starting lineups.")
    else:
        st.warning(
            "No strictly positive-for-both trades survived the current filters. "
            "Showing the closest options instead."
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

    st.caption(
        "The important columns are the two lineup Δ values. They compare each team's "
        "best projected starting lineup before vs. after the trade, averaged across "
        "the projection weeks. Bench-only raw point swaps are not treated as gains."
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