## NFL Game Outcome Prediction (Momentum Only)

A random forest classifier that predicts whether the **home team wins** an NFL game using only **momentum variables** and static game context. It deliberately leaves out betting market information to test how much signal recent team form carries on its own.

### Results (2025 season)

| Approach | Accuracy |
|---|---|
| Always pick the home team | **53.38%** (150 of 281 games) |
| Random forest, momentum variables only | **60%** |
| Random forest with Vegas spread line added | **65.83%** |
| Vegas spread line alone | **65.48%** |

- The model beats the home team baseline by about **5 percentage points** without any betting information.
- It picks the home team in about 53% of games, close to the real home win rate, so it is **not simply defaulting to the majority class**.
- Adding the spread pushes accuracy above 65%, but the spread alone gets nearly the same result. That is why it is excluded from the main model.

### Data

- Source: **nflfastR** play by play data, cleaned and merged into one game level table.
- Training: seasons **2010 to 2024**.
- Test: the **2025 season**, never seen during training.
- Target: `result`, where 1 means the home team won.
- **Season is not a predictor.** It is used only to split train and test, then dropped.

### Features

**Static game variables** (shared by both teams):
- `location`, `div_game`, `temp`, `wind`, `altitude`, `grass`, `week`

**Momentum variables** (computed separately for the home and away team, **50 in total**):
- **Lagged over the past 3 games** (`_lag1`, `_lag2`, `_lag3`, with a `.home` and `.away` version of each):
  - `successRatio`
  - `average_cpoe`
  - `avg_epa_per_play`
  - `thirddownconverted`
  - `thirddownfailed`
  - `fumblelost`
  - `interception`
  - `sack`
- **Not lagged** (one flag per team, `.home` and `.away`):
  - `sameQuarterback`

### Files

- `randomClfMerge.csv`: the cleaned dataset that produced the highest accuracy (58%).
- `prediction.model.py`: trains the random forest on 2010 to 2024 and evaluates on 2025.

### How to run

1. Install dependencies:

       pip install pandas scikit-learn

2. Put the CSV in a `data/` folder next to the script.
3. Run:

       python prediction.model.py

The script prints the success rate, the number of home wins, and the home team only baseline. To see which variables matter most, change `mapImportance(clf, False)` to `True` in the main block.

### Notes

- `season` and `spread_line` are both dropped before training and testing. Keep `spread_line` in to reproduce the 65%+ result.
- `random_state=0` makes results reproducible.
- The split is chronological (train on the past, test on 2025), so no future information leaks into training.
- This is a personal side project and an accuracy benchmark, not a betting tool. A 5 point edge over the home team baseline on one season is a modest result.
