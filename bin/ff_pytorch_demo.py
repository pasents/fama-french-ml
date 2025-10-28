"""
ff_pytorch_demo.py
------------------
PyTorch on Fama–French 5 + Momentum (Europe), with time-series splits.

What it does:
  • Loads your merged FF dataset (monthly or daily)
  • Builds lagged factor features (default: 3 lags)
  • Predicts next-period excess market return (MKT_RF) by default
  • Time-based train/val/test split (70/15/15)
  • Standardization fit on train only
  • Linear baseline or MLP (choose via --model)
  • Metrics: MSE/MAE/R2 + directional accuracy
  • Simple sign-strategy backtest on test set + Sharpe, CAGR
  • Saves best model checkpoint + TorchScript

Run examples:
  python ff_pytorch_demo.py --freq monthly
  python ff_pytorch_demo.py --freq daily --model mlp --lags 6 --epochs 200
  python ff_pytorch_demo.py --path "datasets/europe_ff5_plus_mom_monthly.csv" --target MKT_RF
"""

import argparse, os, json, math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------- Config / defaults ----------------
DEFAULT_MONTHLY = "datasets/europe_ff5_plus_mom_monthly.csv"
DEFAULT_DAILY   = "datasets/europe_ff5_plus_mom_daily.csv"
ANNUALIZE = {"monthly": 12, "daily": 252}

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.strip().upper() for c in df.columns]
    df = df.copy()
    df.columns = cols
    alias = {"MKT-RF":"MKT_RF", "UMD":"MOM", "WML":"MOM"}
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})
    return df

# --------------- Torch bits ----------------
class TabDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class LinearReg(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    def forward(self, x): return self.linear(x)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# --------------- Utilities ----------------
def make_lagged(df: pd.DataFrame, cols: List[str], lags: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for L in range(1, lags+1):
            out[f"{c}_L{L}"] = out[c].shift(L)
    return out

def time_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val
    i1, i2 = n_train, n_train + n_val
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]

@dataclass
class Scaler:
    mean: np.ndarray
    std:  np.ndarray
    def transform(self, X): return (X - self.mean) / np.where(self.std==0, 1.0, self.std)
    @classmethod
    def fit(cls, X):
        return cls(mean=X.mean(0), std=X.std(0))

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def sharpe_ratio(r, periods_per_year):
    mu = r.mean() * periods_per_year
    sig = r.std(ddof=1) * np.sqrt(periods_per_year)
    return 0.0 if sig == 0 else mu / sig

def cagr(returns, periods_per_year):
    nav = (1.0 + returns).cumprod()
    if len(nav) == 0 or nav.iloc[0] <= 0: return 0.0
    n_years = len(returns) / periods_per_year
    return nav.iloc[-1] ** (1.0 / max(n_years, 1e-9)) - 1.0

# --------------- Training / Eval ----------------
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device="cpu"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val, best_state = float("inf"), None

    for ep in range(1, epochs+1):
        # train
        model.train()
        total = 0.0; count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        train_mse = total / max(count,1)

        # val
        model.eval()
        vtotal = 0.0; vcount = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vtotal += loss.item() * xb.size(0)
                vcount += xb.size(0)
        val_mse = vtotal / max(vcount,1)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % max(epochs//10,1) == 0 or ep == 1:
            print(f"epoch {ep:4d} | train MSE={train_mse:.6f} | val MSE={val_mse:.6f}")

    model.load_state_dict(best_state)
    return model, best_val

def evaluate(model, X: np.ndarray, y: np.ndarray, device="cpu"):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy().squeeze(1)
    y_true = y
    mse = np.mean((pred - y_true)**2)
    mae = np.mean(np.abs(pred - y_true))
    # R^2
    ss_res = np.sum((y_true - pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    da = directional_accuracy(y_true, pred)
    return pred, {"mse":mse, "mae":mae, "r2":r2, "dir_acc":da}

# --------------- Main pipeline ----------------
def main():
    p = argparse.ArgumentParser(description="PyTorch on Europe Fama–French + Momentum (predict next-period excess return).")
    p.add_argument("--freq", choices=["monthly","daily"], default="monthly")
    p.add_argument("--path", default=None, help="CSV path; if omitted uses default by freq.")
    p.add_argument("--target", default="MKT_RF", help="Column to predict (default: MKT_RF = excess market return).")
    p.add_argument("--lags", type=int, default=3, help="Number of lags for factor features.")
    p.add_argument("--model", choices=["linear","mlp"], default="mlp")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    csv_path = args.path or (DEFAULT_MONTHLY if args.freq=="monthly" else DEFAULT_DAILY)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = ensure_cols(df)

    # Build MARKET total return if present (optional)
    if "MKT_RF" in df.columns and "RF" in df.columns and "MARKET" not in df.columns:
        df["MARKET"] = df["MKT_RF"] + df["RF"]

    # Factor set (you can edit if your file has different names)
    factor_cols = [c for c in ["MKT_RF","SMB","HML","RMW","CMA","MOM"] if c in df.columns]
    if len(factor_cols) == 0:
        raise ValueError("No factor columns found. Expected some of: MKT_RF, SMB, HML, RMW, CMA, MOM")

    # Target column
    target_col = args.target.upper()
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not in CSV columns: {list(df.columns)}")

    # Create lagged features; target is next-period target_col
    lagged = make_lagged(df[factor_cols], factor_cols, args.lags)
    lagged["Y_NEXT"] = df[target_col].shift(-1)

    # Drop rows with NaNs from lagging/leading
    lagged = lagged.dropna()
    print(f"[INFO] Rows after lagging: {len(lagged)}; features={lagged.shape[1]-1}")

    # Time split (70/15/15)
    train_df, val_df, test_df = time_split(lagged, 0.7, 0.15)

    X_train = train_df.drop(columns=["Y_NEXT"]).values
    y_train = train_df["Y_NEXT"].values
    X_val   = val_df.drop(columns=["Y_NEXT"]).values
    y_val   = val_df["Y_NEXT"].values
    X_test  = test_df.drop(columns=["Y_NEXT"]).values
    y_test  = test_df["Y_NEXT"].values

    # Scale using train only
    scaler = Scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # DataLoaders
    train_loader = DataLoader(TabDS(X_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(TabDS(X_val,   y_val),   batch_size=args.batch, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = X_train.shape[1]
    model = (LinearReg(in_dim) if args.model=="linear" else MLP(in_dim, hidden=64))

    print(f"[INFO] Device: {device} | Model: {args.model} | InDim: {in_dim} | Epochs: {args.epochs}")
    model, best_val = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)
    print(f"[INFO] Best val MSE: {best_val:.8f}")

    # Evaluate on train/val/test
    yhat_tr, tr_metrics = evaluate(model, X_train, y_train, device)
    yhat_vl, vl_metrics = evaluate(model, X_val,   y_val,   device)
    yhat_te, te_metrics = evaluate(model, X_test,  y_test,  device)

    print("\n=== Metrics ===")
    print("Train:", tr_metrics)
    print("Val:  ", vl_metrics)
    print("Test: ", te_metrics)

    # Tiny sign strategy on test (excess returns): long if pred>0 else short
    periods = ANNUALIZE[args.freq]
    strat_ret = np.sign(yhat_te) * y_test
    sr = sharpe_ratio(pd.Series(strat_ret), periods)
    cg = cagr(pd.Series(strat_ret), periods)
    hit = directional_accuracy(y_test, yhat_te)
    print(f"\n=== Sign strategy on test (target = {target_col}) ===")
    print(f"Directional accuracy: {hit:.3f}")
    print(f"Annualized Sharpe:   {sr:.3f}")
    print(f"CAGR:                {cg:.3%}")

    # Save artifacts
    ckpt_name = f"ff_{args.freq}_{args.model}_best.pt"
    torch.save(model.state_dict(), ckpt_name)
    with open(f"ff_{args.freq}_scaler.json","w") as f:
        json.dump({"mean": scaler.mean.tolist(), "std": scaler.std.tolist(), "cols": list(train_df.drop(columns=['Y_NEXT']).columns)}, f)
    # TorchScript
    scripted = torch.jit.script(model.cpu())
    scripted.save(f"ff_{args.freq}_{args.model}_scripted.pt")

    print("\n✅ Saved:")
    print(f"- {ckpt_name} (state_dict)")
    print(f"- ff_{args.freq}_{args.model}_scripted.pt (TorchScript)")
    print(f"- ff_{args.freq}_scaler.json (feature scaling + column order)")

if __name__ == "__main__":
    main()
