import pandas as pd
from pathlib import Path

BASE = Path("Investment_universe")

# 1) Load current universe and data
uni = pd.read_csv(BASE/"modeling_universe.csv")
rets = pd.read_csv(BASE/"modeling_returns.csv", index_col=0)
prices = pd.read_csv(BASE/"modeling_prices.csv", index_col=0)

# 2) Force-drop the two outliers
drop = {"CLI.L","STAN.L"}
uni2 = uni[~uni["Ticker"].isin(drop)]

# 3) Filter datasets
tickers = uni2["Ticker"].tolist()
rets2   = rets[tickers]
prices2 = prices[tickers]

# 4) Save
uni2.to_csv(BASE/"modeling_universe.csv", index=False)
rets2.to_csv(BASE/"modeling_returns.csv")
prices2.to_csv(BASE/"modeling_prices.csv")

print(len(uni), "→", len(uni2), "tickers")
print("returns shape:", rets2.shape, "prices shape:", prices2.shape)
