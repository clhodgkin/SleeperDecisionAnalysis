
# Sleeper Mutual-Gain Trade Finder

A small Python + Streamlit project that:

1. Looks up a Sleeper user.
2. Loads that user's NFL leagues.
3. Pulls league users, rosters, settings, and player metadata.
4. Pulls weekly Sleeper projections.
5. Searches 1-for-1, 1-for-2, 2-for-1, and 2-for-2 trades.
6. Scores each trade by **change in each team's optimized projected starting lineup per week**.
7. Can export a JSON snapshot of the league's public Sleeper data.

## Why lineup impact instead of raw player points?

If a trade is evaluated only by the sum of player projections sent and received, the exchange is essentially zero-sum: one side receives more projected points and the other receives fewer.

Lineup optimization allows a trade to be positive for both teams. Example: Team A has extra RB depth but weak WRs, while Team B has extra WR depth but weak RBs. Swapping comparable assets can raise both teams' actual starting-lineup projection.

## Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Streamlit app

```bash
python -m streamlit run app.py
```

Enter your Sleeper username and season, choose the league, then choose a trade partner.

## Export league data from the command line

First find your league ID in the Sleeper URL or from the Streamlit selector.

```bash
python collect_snapshot.py \
  --username YOUR_SLEEPER_USERNAME \
  --league-id YOUR_LEAGUE_ID \
  --season 2026
```

To embed Sleeper's full NFL player catalog too:

```bash
python collect_snapshot.py \
  --username YOUR_SLEEPER_USERNAME \
  --league-id YOUR_LEAGUE_ID \
  --season 2026 \
  --include-player-catalog
```

## Data collected in the JSON snapshot

- user object
- league metadata/settings/scoring
- league users
- rosters
- matchups, weeks 1–18
- transactions, weeks 1–18
- winners bracket
- losers bracket
- traded draft picks
- league drafts
- draft details
- draft picks
- draft traded picks
- optional full NFL player catalog

## Projection caveat

Sleeper's core league API is officially documented and read-only. The projection feed used here is currently available but is not part of Sleeper's current official API documentation. It is isolated in `SleeperClient.get_week_projections()` so it can be replaced if Sleeper changes it.

The default scoring mode chooses Sleeper's standard PPR / half-PPR / standard projection rollup based on the league's reception scoring.

The "League settings (experimental custom scorer)" mode performs a dot product between projected stat keys and the league's scoring settings. It works for many ordinary custom scoring rules but may not exactly reproduce exotic threshold bonuses, position premiums, or unusual defensive scoring.

## Practical next upgrades

- Include waiver-wire replacement value for 2-for-1 trades.
- Add dynasty draft-pick values.
- Weight future weeks instead of treating all weeks equally.
- Add injury-risk adjustments.
- Add schedule/playoff-week weighting.
- Add "untouchable" players and position-needs controls.
- Generate trades across every opponent automatically rather than one selected opponent at a time.


## Windows: easiest way to launch

This customized build is prefilled with:

- Sleeper username: `chodgkin`
- Preferred league ID: `1393026126246346752` (`Sacks in the City`)

Double-click:

```text
setup_windows.bat
```

the first time. It installs the requirements using the same Python interpreter and launches Streamlit.

After setup, you can normally double-click:

```text
run_app.bat
```

### If PowerShell says `streamlit` is not recognized

That only means the Streamlit executable is not on your Windows PATH. Run it as a Python module instead:

```powershell
python -m streamlit run app.py
```

If that says `No module named streamlit`, install the requirements using the same Python interpreter:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Useful diagnostics:

```powershell
python --version
python -m pip --version
python -m pip show streamlit
```
