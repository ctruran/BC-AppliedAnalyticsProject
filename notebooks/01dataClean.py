from pathlib import Path
import pandas as pd

cwd = Path.cwd().resolve()
REPO = next(p for p in [cwd, *cwd.parents] if (p / ".git").exists())

dev = pd.read_csv(REPO / "data" / "raw" / "DEV _ March Madness.csv")
spell = pd.read_csv(REPO / "data" / "raw" / "MTeamSpellings.csv")
games = pd.read_csv(REPO / "data" / "raw" / "MarchMadnessGameStats2003-2024.csv")

(REPO / "data" / "processed").mkdir(parents=True, exist_ok=True)
(REPO / "data" / "interim").mkdir(parents=True, exist_ok=True)

# TeamID -> team name 
spell["nameKey"] = spell["TeamNameSpelling"].astype(str).str.strip().str.lower()
dev["nameKey"] = dev["Mapped ESPN Team Name"].astype(str).str.strip().str.lower()
dev = dev.merge(spell[["TeamID", "nameKey"]].drop_duplicates(), on="nameKey", how="left")

# team and which season
team_season = (
    dev.dropna(subset=["TeamID"])
       .drop_duplicates(["Season", "TeamID"])
       .copy()
)
if "nameKey" in team_season.columns:
    team_season.drop(columns=["nameKey"], inplace=True)

# tournament outcomes 
wins = (games.groupby(["Season", "WTeamID"]).size()
        .rename("tourney_wins").reset_index().rename(columns={"WTeamID": "TeamID"}))
losses = (games.groupby(["Season", "LTeamID"]).size()
          .rename("tourney_losses").reset_index().rename(columns={"LTeamID": "TeamID"}))

t = wins.merge(losses, on=["Season", "TeamID"], how="outer").fillna(0)
t["tourney_wins"] = t["tourney_wins"].astype(int)
t["tourney_losses"] = t["tourney_losses"].astype(int)
t["tourney_games"] = t["tourney_wins"] + t["tourney_losses"]

def wins_to_round(w):
    if w <= 0: return "R64"
    if w == 1: return "R32"
    if w == 2: return "S16"
    if w == 3: return "E8"
    if w == 4: return "F4"
    if w == 5: return "Final"
    return "Champion"

t["round_reached"] = t["tourney_wins"].map(wins_to_round)

team_season = team_season.merge(t, on=["Season", "TeamID"], how="left")
team_season["tourney_wins"] = team_season["tourney_wins"].fillna(0).astype(int)
team_season["tourney_losses"] = team_season["tourney_losses"].fillna(0).astype(int)
team_season["tourney_games"] = team_season["tourney_games"].fillna(0).astype(int)
team_season["round_reached"] = team_season["round_reached"].fillna("NoTourney")

# only march madness teams allowed
mm = (team_season.loc[team_season["Post-Season Tournament"].eq("March Madness"), ["Season", "TeamID"]]
      .drop_duplicates())

g = games[[c for c in ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"] if c in games.columns]].copy()
g = g.merge(mm.rename(columns={"TeamID": "WTeamID"}), on=["Season", "WTeamID"], how="inner")
g = g.merge(mm.rename(columns={"TeamID": "LTeamID"}), on=["Season", "LTeamID"], how="inner")

g["Team1ID"] = g[["WTeamID", "LTeamID"]].min(axis=1)
g["Team2ID"] = g[["WTeamID", "LTeamID"]].max(axis=1)
g["y"] = (g["Team1ID"] == g["WTeamID"]).astype(int)
g["WinnerTeamID"] = g["WTeamID"]

# convert id to names 
names = (team_season[["Season", "TeamID", "Mapped ESPN Team Name"]]
         .drop_duplicates(["Season", "TeamID"])
         .rename(columns={"Mapped ESPN Team Name": "TeamName"}))
g = g.merge(names.rename(columns={"TeamID": "Team1ID", "TeamName": "Team1Name"}),
            on=["Season", "Team1ID"], how="left")
g = g.merge(names.rename(columns={"TeamID": "Team2ID", "TeamName": "Team2Name"}),
            on=["Season", "Team2ID"], how="left")

# features
t1 = team_season.rename(columns={"TeamID": "Team1ID"}).copy()
t1 = t1.rename(columns={c: f"team1_{c}" for c in t1.columns if c not in ["Season", "Team1ID"]})
t2 = team_season.rename(columns={"TeamID": "Team2ID"}).copy()
t2 = t2.rename(columns={c: f"team2_{c}" for c in t2.columns if c not in ["Season", "Team2ID"]})

tourney_games = g.merge(t1, on=["Season", "Team1ID"], how="left").merge(
    t2, on=["Season", "Team2ID"], how="left"
)


# drops columns with more than 30% of rows missing the data.
# https://stackoverflow.com/questions/60450808/remove-columns-with-missing-values-above-a-threshold-pandas
key = {
    "Season", "DayNum", "Team1ID", "Team2ID", "WinnerTeamID", "y",
    "WTeamID", "LTeamID", "WScore", "LScore", "Team1Name", "Team2Name",
    "team1_Post-Season Tournament", "team2_Post-Season Tournament"
}
drop_cols = [c for c, frac in tourney_games.isna().mean().items() if frac > 0.30 and c not in key]
tourney_games = tourney_games.drop(columns=drop_cols)

team_season.to_csv(REPO / "data" / "processed" / "teamSeason.csv", index=False)
tourney_games.to_csv(REPO / "data" / "interim" / "tourneyGames.csv", index=False)

print("Saved:")
print(REPO / "data" / "processed" / "teamSeason.csv")
print(REPO / "data" / "interim" / "tourneyGames.csv")
print("Rows (games):", len(tourney_games))
print("Dropped cols (>30% missing):", len(drop_cols))
