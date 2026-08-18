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

BUY_SCORE = 80
WATCH_SCORE = 60
HIGH_NEAR = 0.02
RSI_LOW = 35
RSI_HIGH = 65

RETRIES = 3
RETRY_DELAY = 2

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
    # กรองค่า NaN หรือ None ก่อนเซฟลง JSON
    clean_state = {}
    for k, v in state.items():
        clean_state[k] = v.copy()
        price = clean_state[k].get("last_price")
        if price is None or pd.isna(price) or np.isnan(price):
            clean_state[k]["last_price"] = None
        else:
            clean_state[k]["last_price"] = float(price)

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_state, f, indent=2, ensure_ascii=False)

def rsi(close, period=14):
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

def fetch(ticker):
    try:
        # ใช้ yf.Ticker ดึง history โดยตรง ป้องกันปัญหาสับสน Column MultiIndex
        t = yf.Ticker(ticker)
        df = t.history(period="3y", interval="1d", auto_adjust=True)
        
        if df is None or df.empty or len(df) < 260:
            print(f"{ticker}: insufficient price history, skipping")
            return None

        # บังคับดึง Series แบบ 1D แน่นอน
        close = df["Close"].squeeze().dropna()
        volume = df["Volume"].squeeze().fillna(0)

        if len(close) < 260:
            return None

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        high52 = close.rolling(252).max()
        vol20 = volume.rolling(20).mean()
        r = rsi(close)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        info = {}
        try:
            info = t.info or {}
        except Exception as e:
            print(f"{ticker}: fundamentals unavailable: {e}")

        last_val = close.iloc[-1]
        if pd.isna(last_val):
            return None

        return {
            "price": float(last_val),
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
    raw = 0
    max_possible = 0
    reasons = []
    risk_flags = []

    # Trend: 25 pts
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

    # Momentum: 20 pts
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

    # Breakout / volume: 20 pts
    max_possible += 20
    near_high = (d["high52"] - d["price"]) / d["high52"] <= HIGH_NEAR
    if near_high:
        raw += 10
        reasons.append("ราคาอยู่ใกล้ 52W High")

    if d["vol20"] > 0 and d["volume"] > d["vol20"] * 1.2:
        raw += 10
        reasons.append("Volume สูงกว่าค่าเฉลี่ย 20 วัน >20%")

    # Fundamentals: up to 25 pts
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

    if d["price"] < d["ma200"]:
        risk_flags.append("ราคาอยู่ต่ำกว่า MA200")
    if d["rsi"] > 70:
        risk_flags.append("RSI >70")
    if d["vol20"] > 0 and d["volume"] < d["vol20"] * 0.6:
        risk_flags.append("Volume ต่ำ")

    normalized_score = round((raw / max_possible) * 100) if max_possible else 0
    fundamentals_missing = max_possible < 65 + 25

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
    vol_ratio = (d['volume']/d['vol20']) if d['vol20'] > 0 else 0
    msg = (
        f"{emoji} {ticker} — {label}\n\n"
        f"คะแนน: {score_value}/100\n"
        f"ราคา: ${d['price']:.2f}\n"
        f"MA20: ${d['ma20']:.2f}\n"
        f"MA50: ${d['ma50']:.2f}\n"
        f"MA200: ${d['ma200']:.2f}\n"
        f"52W High: ${d['high52']:.2f}\n"
        f"RSI: {d['rsi']:.1f}\n"
        f"Volume/Avg20: {vol_ratio:.2f}x\n\n"
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

        old = state.get(ticker, {}).get("label")
        if label == "BUY CANDIDATE" and old != "BUY CANDIDATE":
            send_telegram(make_message(ticker, d, s, label, reasons, risks))

        state[ticker] = {
            "label": label,
            "score": s,
            "last_price": d["price"],
            "updated": datetime.now(timezone.utc).isoformat()
        }
        time.sleep(0.5)

    save_state(state)

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n=== TOP SIGNALS ===")
    for ticker, s, label, d in results[:10]:
        print(f"{ticker:5} {s:3}/100  {label:15} ${d['price']:.2f}")

    if skipped:
        print(f"\nSkipped {len(skipped)} ticker(s) due to data errors: {', '.join(skipped)}")

if __name__ == "__main__":
    main()
