import pandas as pd
import numpy as np

def load_series(path, colname):
    df = pd.read_csv(path)
    df = df.melt(id_vars="Year", var_name="Month", value_name=colname)
    df = df.dropna(subset=[colname])
    month_map = {m: i+1 for i, m in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}
    df["month_num"] = df["Month"].map(month_map)
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["month_num"], day=1))
    df = df[["date", colname]].sort_values("date").reset_index(drop=True)
    return df

base = "/home/claude/svd_report/data/"
c1 = load_series(base + "raw_c1.csv", "C1")
c2 = load_series(base + "raw_c2.csv", "C2")
c3 = load_series(base + "raw_c3.csv", "C3")
c4 = load_series(base + "raw_c4.csv", "C4")

panel = c1.merge(c2, on="date").merge(c3, on="date").merge(c4, on="date")
panel = panel.sort_values("date").reset_index(drop=True)

print("Panel shape:", panel.shape)
print(panel.head())
print(panel.tail())
print("Any NaN:", panel.isna().sum().sum())

panel.to_csv("/home/claude/svd_report/data/wti_curve_panel.csv", index=False)
print("Saved.")
