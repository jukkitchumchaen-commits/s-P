import yfinance as yf
import pandas as pd
import numpy as np

# Backtest over the same watchlist used by the live bot by default.
# Set TICKERS to a single-item list to test one stock only.
TICKERS = [
    "NVDA","AAPL","MSFT","GOOGL","META","TSLA","AMD","AVGO","NFLX",
    "LLY","JNJ","PFE","ABBV","BA","LMT","RTX","NOC",
    "JPM","V","MA","PYPL","WMT","COST","KO","PEP","XOM","CVX"
]
PERIOD = "10y"

# NOTE ON SCOPE: this backtest is technical-only (trend / momentum /
# breakout-volume). It intentionally excludes fundamentals (EPS growth,
# revenue growth, forward P/E) because yf.Ticker().info only exposes
# CURRENT fundamentals, not point-in-time historical values. Scoring
# historical bars with today's fundamentals would introduce look-ahead
# bias and make results unusable. stock_bot.py (live) additionally scores
# fundamentals when available; this backtest validates the technical
# component only, on its own normalized 0-100 scale.
BUY_SCORE = 80
TRANSACTION_COST_PCT = 0.15  # per leg (entry + exit), in percent, e.g. commission+slippage


def rsi(close, period=14):
    """Same fixed RSI as stock_bot.py: avg_loss == 0 -> RSI 100, not NaN."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rsi_val = pd.Series(index=close.index, dtype=float)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    loss_zero = (avg_loss == 0) & (avg_gain != 0)
    normal = ~both_zero & ~loss_zero

    rs = avg_gain[normal] / avg_loss[normal]
    rsi_val[normal] = 100 - (100 / (1 + rs))
    rsi_val[loss_zero] = 100.0
    rsi_val[both_zero] = 50.0
    return rsi_val


def precompute_indicators(df):
    """Vectorized precompute so we don't recompute rolling windows in an O(n^2) loop."""
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    ind = pd.DataFrame(index=df.index)
    ind["price"] = close
    ind["ma20"] = close.rolling(20).mean()
    ind["ma50"] = close.rolling(50).mean()
    ind["ma200"] = close.rolling(200).mean()
    ind["high52"] = close.rolling(252).max()
    ind["rsi"] = rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ind["macd"] = ema12 - ema26
    ind["macd_signal"] = ind["macd"].ewm(span=9, adjust=False).mean()

    ind["volume"] = volume
    ind["vol20"] = volume.rolling(20).mean()
    return ind


def calc_score(row):
    """Technical-only score, normalized 0-100. Mirrors the technical portion
    of score() in stock_bot.py (trend 25 + momentum 20 + breakout/volume 20
    = 65 raw points, scaled to 100)."""
    if pd.isna(row["ma200"]) or pd.isna(row["high52"]) or pd.isna(row["rsi"]):
        return np.nan

    raw = 0
    raw += 10 if row["price"] > row["ma200"] else 0
    raw += 8 if row["ma50"] > row["ma200"] else 0
    raw += 7 if row["ma20"] > row["ma50"] else 0
    raw += 10 if 45 <= row["rsi"] <= 65 else (5 if row["rsi"] < 35 else 0)
    raw += 10 if row["macd"] > row["macd_signal"] else 0
    raw += 10 if (row["high52"] - row["price"]) / row["high52"] <= 0.02 else 0
    raw += 10 if row["volume"] > row["vol20"] * 1.2 else 0

    return round((raw / 65) * 100)


def run_single(ticker):
    df = yf.download(ticker, period=PERIOD, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 260:
        print(f"{ticker}: insufficient history, skipping")
        return pd.DataFrame()

    ind = precompute_indicators(df)
    ind["score"] = ind.apply(calc_score, axis=1)

    trades = []
    in_position = False
    entry = None
    entry_date = None

    for i in range(252, len(ind)):
        price = ind["price"].iloc[i]
        ma200 = ind["ma200"].iloc[i]
        s = ind["score"].iloc[i]

        if pd.isna(s) or pd.isna(ma200):
            continue

        if not in_position and s >= BUY_SCORE:
            in_position = True
            entry = price
            entry_date = ind.index[i]
        elif in_position and price < ma200:
            gross_ret = (price / entry - 1) * 100
            net_ret = gross_ret - (2 * TRANSACTION_COST_PCT)  # entry + exit legs
            trades.append((ticker, entry_date, ind.index[i], entry, price, gross_ret, net_ret))
            in_position = False

    if in_position:
        price = ind["price"].iloc[-1]
        gross_ret = (price / entry - 1) * 100
        net_ret = gross_ret - (2 * TRANSACTION_COST_PCT)
        trades.append((ticker, entry_date, ind.index[-1], entry, price, gross_ret, net_ret))

    return pd.DataFrame(
        trades,
        columns=["ticker", "entry_date", "exit_date", "entry_price", "exit_price", "gross_return_pct", "net_return_pct"]
    )


def run():
    all_trades = []
    for ticker in TICKERS:
        print(f"Backtesting {ticker}...")
        t = run_single(ticker)
        if not t.empty:
            all_trades.append(t)

    if not all_trades:
        print("No trades found across watchlist.")
        return

    out = pd.concat(all_trades, ignore_index=True)
    print("\n" + out.to_string(index=False))

    print("\n=== AGGREGATE STATS (net of est. transaction costs) ===")
    print("Total trades:", len(out))
    print("Win rate:", round((out["net_return_pct"] > 0).mean() * 100, 2), "%")
    print("Average return/trade:", round(out["net_return_pct"].mean(), 2), "%")
    print("Median return/trade:", round(out["net_return_pct"].median(), 2), "%")
    print("Best trade:", round(out["net_return_pct"].max(), 2), "%")
    print("Worst trade:", round(out["net_return_pct"].min(), 2), "%")

    print("\n=== PER-TICKER SUMMARY ===")
    summary = out.groupby("ticker")["net_return_pct"].agg(["count", "mean", "median"])
    summary.columns = ["trades", "avg_return_pct", "median_return_pct"]
    print(summary.sort_values("avg_return_pct", ascending=False).to_string())

    print(
        "\nNote: this is a simplified single-position-per-ticker backtest with no "
        "portfolio-level position sizing, no slippage modeling beyond a flat "
        "transaction cost estimate, and technical-only scoring. It is a research "
        "tool, not a guarantee of future performance."
    )


if __name__ == "__main__":
    run()
