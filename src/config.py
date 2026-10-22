"""Shared paths, constants and name-normalisation maps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "ipl_csv"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
DEMO = ROOT / "demo"

# ---- split definition (by season, never by ball) -------------------------------------
TRAIN_END = 2023          # train: seasons <= 2023
VAL_SEASONS = [2024]      # validate / tune / calibrate on 2024
TEST_SEASONS = [2025, 2026]  # touched once, at the very end
IMPACT_PLAYER_FIRST_SEASON = 2023

# ---- cricket constants ----------------------------------------------------------------
BALLS_PER_INNINGS = 120
POWERPLAY_END = 36   # balls bowled before ball 37 -> over 1-6 are powerplay
DEATH_START = 90     # overs 16-20 are the death overs

# Franchise renames: map every historical spelling to the current franchise name.
TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Venue names in Cricsheet vary by era ("Wankhede Stadium" vs "Wankhede Stadium, Mumbai").
# Rule: drop everything after the first comma, then apply these aliases for true renames.
VENUE_ALIAS = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",            # same ground, renamed 2019
    "Sardar Patel Stadium": "Narendra Modi Stadium",       # same ground, renamed 2021
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium",
    "Zayed Cricket Stadium": "Zayed Cricket Stadium",
}


def norm_team(t: str) -> str:
    return TEAM_MAP.get(t, t)


def norm_venue(v: str) -> str:
    base = v.split(",")[0].strip()
    return VENUE_ALIAS.get(base, base)

ABBR = {
    "Mumbai Indians": "MI", "Chennai Super Kings": "CSK", "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR", "Delhi Capitals": "DC", "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR", "Sunrisers Hyderabad": "SRH", "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG", "Deccan Chargers": "DEC", "Pune Warriors": "PWI",
    "Gujarat Lions": "GL", "Rising Pune Supergiant": "RPS", "Kochi Tuskers Kerala": "KTK",
}

# Run-rate smoothing: raw runs/balls after 1-2 balls is absurd (a first-ball six -> "36 runs per over"),
# and a fixed prior (we first tried 7.8 rpo) makes every innings START "below par" once scoring drifts
# upward (2025-26 average ~9.6 rpo). So we shrink the current run rate toward the *recent league scoring
# rate* (mean first-innings score of the previous 40 matches / 20 overs), as if CRR_PRIOR_BALLS balls had
# already been bowled at that rate. At ball 0 the projected score therefore equals the league average
# (DECISIONS #13).
CRR_PRIOR_BALLS = 18
