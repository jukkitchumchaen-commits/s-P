import os, json, time
from datetime import datetime, timezone
import requests
import yfinance as yf
import pandas as pd
import numpy as np

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = "alert_state.json"

WATCHLIST = [
    "NVDA","AAPL","MSFT","GOOGL","META","TSLA","AMD","AVGO","NFLX",
    "LLY","JNJ","PFE","ABBV","BA","LMT","RTX","NOC",
    "JPM","V","MA","PYPL","WMT","COST","KO","PEP","XOM","CVX"
]

# Signal thresholds (normalized 0-100 scale, see score())
BUY_SCORE = 80
WATCH_SCORE = 60
HIGH_NEAR = 0.02
RSI_LOW = 35
RSI_HIGH = 65

RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles each retry


def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        print("Telegram credentials are missing.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print("Telegram error:", e)


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def rsi(close, period=14):
    """
    Wilder-style RSI via EWM.
    FIX: when avg_loss == 0 (no down days in the lookback window), RS is
    infinite and RSI should be 100, not NaN. Previously the code replaced
    0 with NaN, silently dropping any stock on a pure uptrend from scoring.
    """
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


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def with_retries(fn, *args, **kwargs):
    """Retry wrapper for flaky Yahoo Finance calls, with exponential backoff."""
    delay = RETRY_DELAY
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt}/{RETRIES} failed: {e}")
            if attempt < RETRIES:
                time.sleep(delay)
                delay *= 2
    raise last_err


def fetch(ticker):
    try:
        df = with_retries(
            yf.download, ticker, period="3y", interval="1d",
            auto_adjust=True, progress=False, threads=False
        )
        df = flatten_columns(df)
        if df is None or df.empty or len(df) < 260:
            print(f"{ticker}: insufficient price history, skipping")
            return None

        # Sanity check: most recent bar should not be stale (e.g. > 5 calendar
        # days old), which can happen if Yahoo returns a delayed/short dataset.
        last_bar_date = df.index[-1]
        if hasattr(last_bar_date, "to_pydatetime"):
            age_days = (datetime.now(timezone.utc) - last_bar_date.to_pydatetime().replace(tzinfo=timezone.utc)).days
            if age_days > 5:
                print(f"{ticker}: latest bar is {age_days} days old, data may be stale")

        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        high52 = close.rolling(252).max()
        vol20 = volume.rolling(20).mean()
        r = rsi(close)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        # Approximate fundamentals, when Yahoo provides them.
        # NOTE: yfinance's `.info` fields (trailingPE, earningsGrowth,
        # revenueGrowth) are known to be inconsistent or missing, especially
        # outside US large caps. Treated as optional signal, never required.
        info = {}
        try:
            info = with_retries(lambda: yf.Ticker(ticker).info)
        except Exception as e:
            print(f"{ticker}: fundamentals unavailable: {e}")

        last = float(close.iloc[-1])
        return {
            "price": last,
            "ma20": float(ma20.iloc[-1]),
            "ma50": float(ma50.iloc[-1]),
            "ma200": float(ma200.iloc[-1]),
            "high52": float(high52.iloc[-1]),
            "rsi": float(r.iloc[-1]),
            "volume": float(volume.iloc[-1]),
            "vol20": float(vol20.iloc[-1]),
            "macd": float(macd.iloc[-1]),
            "macd_signal": float(signal.iloc[-1]),
            "prev_close": float(close.iloc[-2]),
            "prev_high52": float(high52.iloc[-2]),
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
        }
    except Exception as e:
        print(f"{ticker}: data error: {e}")
        return None


def score(d):
    """
    Score a stock 0-100.

    FIX: previously fundamentals were worth 25 raw points out of a fixed
    100-point denominator. Since yfinance often lacks eps_growth /
    revenue_growth / forward_pe, most stocks lost up to 25 points for
    missing data rather than for being weak candidates, making BUY_SCORE=80
    nearly unreachable for anything without full fundamentals coverage.

    Now the score is normalized against the maximum number of points that
    were actually available for THIS stock (technical points always count;
    fundamental points only count toward the denominator when the
    underlying data exists), then scaled back to 0-100. This keeps the
    scale comparable across stocks regardless of data completeness.
    """
    raw = 0
    max_possible = 0
    reasons = []
    risk_flags = []

    # --- Trend: 25 points, always available ---
    max_possible += 25
    if d["price"] > d["ma200"]:
        raw += 10
        reasons.append("ราคาอยู่เหนือ MA200")
    if d["ma50"] > d["ma200"]:
        raw += 8
        reasons.append("MA50 อยู่เหนือ MA200")
    if d["ma20"] > d["ma50"]:
        raw += 7
        reasons.append("MA20 อยู่เหนือ MA50")

    # --- Momentum: 20 points, always available ---
    max_possible += 20
    if 45 <= d["rsi"] <= 65:
        raw += 10
        reasons.append(f"RSI {d['rsi']:.1f} อยู่ในโซนโมเมนตัม")
    elif d["rsi"] < RSI_LOW:
        raw += 5
        reasons.append(f"RSI {d['rsi']:.1f} ต่ำ มีโอกาสรีบาวด์แต่เสี่ยง")
    elif d["rsi"] > RSI_HIGH:
        risk_flags.append(f"RSI {d['rsi']:.1f} สูง ระวังแรงขาย")

    if d["macd"] > d["macd_signal"]:
        raw += 10
        reasons.append("MACD อยู่เหนือ Signal")

    # --- Breakout / volume: 20 points, always available ---
    max_possible += 20
    near_high = (d["high52"] - d["price"]) / d["high52"] <= HIGH_NEAR
    if near_high:
        raw += 10
        reasons.append("ราคาอยู่ใกล้ 52W High")

    if d["volume"] > d["vol20"] * 1.2:
        raw += 10
        reasons.append("Volume สูงกว่าค่าเฉลี่ย 20 วัน >20%")

    # --- Fundamentals: up to 25 points, only counted when data exists ---
    if d["eps_growth"] is not None:
        max_possible += 10
        if d["eps_growth"] > 0.10:
            raw += 10
            reasons.append("EPS Growth >10%")
        elif d["eps_growth"] > 0:
            raw += 5
            reasons.append("EPS Growth เป็นบวก")

    if d["revenue_growth"] is not None:
        max_possible += 10
        if d["revenue_growth"] > 0.10:
            raw += 10
            reasons.append("Revenue Growth >10%")
        elif d["revenue_growth"] > 0:
            raw += 5
            reasons.append("Revenue Growth เป็นบวก")

    if d["forward_pe"] is not None and d["pe"] is not None:
        max_possible += 5
        if d["forward_pe"] < d["pe"]:
            raw += 5
            reasons.append("Forward P/E ต่ำกว่า Trailing P/E")

    # Risk penalty (informational, not subtracted from score)
    if d["price"] < d["ma200"]:
        risk_flags.append("ราคาอยู่ต่ำกว่า MA200")
    if d["rsi"] > 70:
        risk_flags.append("RSI >70")
    if d["volume"] < d["vol20"] * 0.6:
        risk_flags.append("Volume ต่ำ")

    normalized_score = round((raw / max_possible) * 100) if max_possible else 0
    fundamentals_missing = max_possible < 65 + 25  # not all fundamental fields present

    if normalized_score >= BUY_SCORE:
        label = "BUY CANDIDATE"
    elif normalized_score >= WATCH_SCORE:
        label = "WATCH"
    else:
        label = "AVOID / WAIT"

    if fundamentals_missing:
        risk_flags.append("ข้อมูล fundamentals ไม่ครบ คะแนนคิดจาก technical เป็นหลัก")

    return normalized_score, label, reasons, risk_flags


def make_message(ticker, d, score_value, label, reasons, risks):
    emoji = "🟢" if label == "BUY CANDIDATE" else ("🟡" if label == "WATCH" else "🔴")
    msg = (
        f"{emoji} {ticker} — {label}\n\n"
        f"คะแนน: {score_value}/100\n"
        f"ราคา: ${d['price']:.2f}\n"
        f"MA20: ${d['ma20']:.2f}\n"
        f"MA50: ${d['ma50']:.2f}\n"
        f"MA200: ${d['ma200']:.2f}\n"
        f"52W High: ${d['high52']:.2f}\n"
        f"RSI: {d['rsi']:.1f}\n"
        f"Volume/Avg20: {d['volume']/d['vol20']:.2f}x\n\n"
        "เหตุผล:\n" + "\n".join("• " + x for x in reasons[:8])
    )
    if risks:
        msg += "\n\n⚠️ ความเสี่ยง:\n" + "\n".join("• " + x for x in risks[:5])
    msg += (
        "\n\n⚠️ สัญญาณจากระบบ ไม่ใช่การรับประกันกำไร "
        "และไม่ใช่คำแนะนำให้ซื้อขายโดยอัตโนมัติ"
    )
    return msg


def main():
    state = load_state()
    results = []
    skipped = []

    print(f"Checking {len(WATCHLIST)} stocks...")
    for ticker in WATCHLIST:
        d = fetch(ticker)
        if not d:
            skipped.append(ticker)
            continue

        s, label, reasons, risks = score(d)
        results.append((ticker, s, label, d))

        # Notify only when a stock enters BUY CANDIDATE.
        old = state.get(ticker, {}).get("label")
        if label == "BUY CANDIDATE" and old != "BUY CANDIDATE":
            send_telegram(make_message(ticker, d, s, label, reasons, risks))

        state[ticker] = {
            "label": label,
            "score": s,
            "last_price": d["price"],
            "updated": datetime.now(timezone.utc).isoformat()
        }
        time.sleep(0.3)

    save_state(state)

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n=== TOP SIGNALS ===")
    for ticker, s, label, d in results[:10]:
        print(f"{ticker:5} {s:3}/100  {label:15} ${d['price']:.2f}")

    if skipped:
        print(f"\nSkipped {len(skipped)} ticker(s) due to data errors: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
