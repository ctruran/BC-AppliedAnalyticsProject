from pathlib import Path
import pandas as pd

cwd = Path.cwd().resolve()
repo = next(p for p in [cwd, *cwd.parents] if (p / ".git").exists())

dev = pd.read_csv(repo / "data" / "raw" / "DEV _ March Madness.csv")
spell = pd.read_csv(repo / "data" / "raw" / "MTeamSpellings.csv")
games = pd.read_csv(repo / "data" / "raw" / "MarchMadnessGameStats2003-2024.csv")

(repo / "data" / "processed").mkdir(parents=True, exist_ok=True)
(repo / "data" / "interim").mkdir(parents=True, exist_ok=True)

# TeamID -> team name
spell["nameKey"] = spell["TeamNameSpelling"].astype(str).str.strip().str.lower()
dev["nameKey"] = dev["Mapped ESPN Team Name"].astype(str).str.strip().str.lower()
dev = dev.merge(spell[["TeamID", "nameKey"]].drop_duplicates(), on="nameKey", how="left")

# team and which season
teamSeason = (
    dev.dropna(subset=["TeamID"])
       .drop_duplicates(["Season", "TeamID"])
       .copy()
)
if "nameKey" in teamSeason.columns:
    teamSeason.drop(columns=["nameKey"], inplace=True)

# tournament outcomes
wins = (games.groupby(["Season", "WTeamID"]).size()
        .rename("tourneyWins").reset_index().rename(columns={"WTeamID": "TeamID"}))
losses = (games.groupby(["Season", "LTeamID"]).size()
          .rename("tourneyLosses").reset_index().rename(columns={"LTeamID": "TeamID"}))

t = wins.merge(losses, on=["Season", "TeamID"], how="outer").fillna(0)
t["tourneyWins"] = t["tourneyWins"].astype(int)
t["tourneyLosses"] = t["tourneyLosses"].astype(int)
t["tourneyGames"] = t["tourneyWins"] + t["tourneyLosses"]

def winsToRound(w):
    if w <= 0: return "R64"
    if w == 1: return "R32"
    if w == 2: return "S16"
    if w == 3: return "E8"
    if w == 4: return "F4"
    if w == 5: return "Final"
    return "Champion"

t["roundReached"] = t["tourneyWins"].map(winsToRound)

teamSeason = teamSeason.merge(t, on=["Season", "TeamID"], how="left")
teamSeason["tourneyWins"] = teamSeason["tourneyWins"].fillna(0).astype(int)
teamSeason["tourneyLosses"] = teamSeason["tourneyLosses"].fillna(0).astype(int)
teamSeason["tourneyGames"] = teamSeason["tourneyGames"].fillna(0).astype(int)
teamSeason["roundReached"] = teamSeason["roundReached"].fillna("NoTourney")

# only march madness teams allowed
mm = (teamSeason.loc[teamSeason["Post-Season Tournament"].eq("March Madness"), ["Season", "TeamID"]]
      .drop_duplicates())

g = games[[c for c in ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"] if c in games.columns]].copy()
g = g.merge(mm.rename(columns={"TeamID": "WTeamID"}), on=["Season", "WTeamID"], how="inner")
g = g.merge(mm.rename(columns={"TeamID": "LTeamID"}), on=["Season", "LTeamID"], how="inner")

g["team1Id"] = g[["WTeamID", "LTeamID"]].min(axis=1)
g["team2Id"] = g[["WTeamID", "LTeamID"]].max(axis=1)
g["y"] = (g["team1Id"] == g["WTeamID"]).astype(int)
g["winnerTeamId"] = g["WTeamID"]

# convert id to names
names = (teamSeason[["Season", "TeamID", "Mapped ESPN Team Name"]]
         .drop_duplicates(["Season", "TeamID"])
         .rename(columns={"Mapped ESPN Team Name": "teamName"}))
g = g.merge(names.rename(columns={"TeamID": "team1Id", "teamName": "team1Name"}),
            on=["Season", "team1Id"], how="left")
g = g.merge(names.rename(columns={"TeamID": "team2Id", "teamName": "team2Name"}),
            on=["Season", "team2Id"], how="left")

# features
t1 = teamSeason.rename(columns={"TeamID": "team1Id"}).copy()
t1 = t1.rename(columns={c: f"team1_{c}" for c in t1.columns if c not in ["Season", "team1Id"]})

t2 = teamSeason.rename(columns={"TeamID": "team2Id"}).copy()
t2 = t2.rename(columns={c: f"team2_{c}" for c in t2.columns if c not in ["Season", "team2Id"]})

tourneyGames = g.merge(t1, on=["Season", "team1Id"], how="left").merge(
    t2, on=["Season", "team2Id"], how="left"
)

# drops columns with more than 30% of rows missing the data.
# https://stackoverflow.com/questions/60450808/remove-columns-with-missing-values-above-a-threshold-pandas
keyCols = {
    "Season", "DayNum", "team1Id", "team2Id", "winnerTeamId", "y",
    "WTeamID", "LTeamID", "WScore", "LScore", "team1Name", "team2Name",
    "team1_Post-Season Tournament", "team2_Post-Season Tournament"
}
dropCols = [c for c, frac in tourneyGames.isna().mean().items() if frac > 0.30 and c not in keyCols]
tourneyGames = tourneyGames.drop(columns=dropCols)

teamSeason.to_csv(repo / "data" / "processed" / "teamSeason.csv", index=False)
tourneyGames.to_csv(repo / "data" / "interim" / "tourneyGames.csv", index=False)
