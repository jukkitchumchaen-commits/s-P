import os
import json
import time
from datetime import datetime, timezone
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = "alert_state.json"

WATCHLIST = [
    # --- Watchlist เดิม (27 ตัว) ---
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMD", "AVGO", "NFLX",
    "LLY", "JNJ", "PFE", "ABBV", "BA", "LMT", "RTX", "NOC",
    "JPM", "V", "MA", "PYPL", "WMT", "COST", "KO", "PEP", "XOM", "CVX",

    # --- Semiconductor & Hardware (8 ตัว) ---
    "TSM", "ASML", "MU", "QCOM", "AMAT", "LRCX", "ARM", "SMCI",

    # --- Cloud, SaaS & AI Software (8 ตัว) ---
    "PLTR", "AMZN", "ORCL", "CRM", "NOW", "SNOW", "ADBE", "MDB",

    # --- Cybersecurity (3 ตัว) ---
    "PANW", "CRWD", "FTNT",

    # --- Fintech, AdTech & Platforms (4 ตัว) ---
    "SQ", "SHOP", "TTD", "UBER"
]

BUY_SCORE = 80
WATCH_SCORE = 60
HIGH_NEAR = 0.02       # 2% จาก 52W High
LEVEL_NEAR_PCT = 0.015 # 1.5% ถือว่าเข้าใกล้แนวรับ/แนวต้าน
RSI_LOW = 35
RSI_HIGH = 65

RETRIES = 3
RETRY_DELAY = 2


# ==========================================
# 2. TELEGRAM & STATE MANAGEMENT
# ==========================================
def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        print("Telegram credentials are missing. Skipping send.")
        print("--- MSG PREVIEW ---")
        print(text)
        print("-------------------")
        return
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
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


# ==========================================
# 3. INDICATORS & CALCULATIONS
# ==========================================
def rsi(close, period=14):
    """Wilder RSI - แก้ไข Edge case เมื่อ avg_loss == 0 ให้ได้ 100 ไม่เป็น NaN"""
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

def calculate_pivot_levels(df):
    """คำนวณ Pivot Points (S1, S2, R1, R2) จากแท่งก่อนหน้า"""
    prev_high = float(df["High"].iloc[-2])
    prev_low = float(df["Low"].iloc[-2])
    prev_close = float(df["Close"].iloc[-2])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = (2 * pivot) - prev_low
    s1 = (2 * pivot) - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    return {"s1": s1, "s2": s2, "r1": r1, "r2": r2}


# ==========================================
# 4. DATA FETCHING (WITH RETRY)
# ==========================================
def fetch(ticker):
    for attempt in range(1, RETRIES + 1):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="3y", interval="1d", auto_adjust=True)
            
            if df is None or df.empty or len(df) < 260:
                print(f"[{ticker}] Data incomplete, skipping.")
                return None

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

            pivots = calculate_pivot_levels(df)

            info = {}
            try:
                info = t.info or {}
            except Exception:
                pass

            last_val = close.iloc[-1]
            prev_val = close.iloc[-2]

            return {
                "price": float(last_val),
                "prev_price": float(prev_val),
                "ma20": float(ma20.iloc[-1]),
                "ma50": float(ma50.iloc[-1]),
                "ma200": float(ma200.iloc[-1]),
                "high52": float(high52.iloc[-1]),
                "rsi": float(r.iloc[-1]),
                "volume": float(volume.iloc[-1]),
                "vol20": float(vol20.iloc[-1]),
                "macd": float(macd.iloc[-1]),
                "macd_signal": float(signal.iloc[-1]),
                "pivots": pivots,
                "pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps_growth": info.get("earningsGrowth"),
                "revenue_growth": info.get("revenueGrowth"),
            }
        except Exception as e:
            print(f"[{ticker}] Attempt {attempt} failed: {e}")
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return None


# ==========================================
# 5. DYNAMIC SCORING SYSTEM
# ==========================================
def score(d):
    raw = 0
    max_possible = 0
    reasons, risk_flags = [], []

    # --- Trend Score (Max 25) ---
    max_possible += 25
    if d["price"] > d["ma200"]: 
        raw += 10
        reasons.append("ราคาอยู่เหนือ MA200")
    if d["ma50"] > d["ma200"]: 
        raw += 8
        reasons.append("MA50 อยู่เหนือ MA200 (Golden Cross)")
    if d["ma20"] > d["ma50"]: 
        raw += 7
        reasons.append("MA20 อยู่เหนือ MA50")

    # --- Momentum Score (Max 20) ---
    max_possible += 20
    if 45 <= d["rsi"] <= 65: 
        raw += 10
        reasons.append(f"RSI {d['rsi']:.1f} อยู่ในโซน Bullish Momentum")
    elif d["rsi"] < RSI_LOW: 
        raw += 5
        reasons.append(f"RSI {d['rsi']:.1f} โซน Oversold")
    elif d["rsi"] > RSI_HIGH: 
        risk_flags.append(f"RSI {d['rsi']:.1f} เริ่มเข้าเขต Overbought")

    if d["macd"] > d["macd_signal"]: 
        raw += 10
        reasons.append("MACD > Signal Line")

    # --- Breakout & Volume Score (Max 20) ---
    max_possible += 20
    if (d["high52"] - d["price"]) / d["high52"] <= HIGH_NEAR: 
        raw += 10
        reasons.append("ราคาใกล้จุดสูงสุดในรอบ 52 สัปดาห์")
    if d["vol20"] > 0 and d["volume"] > d["vol20"] * 1.2: 
        raw += 10
        reasons.append("Volume สูงกว่าค่าเฉลี่ย 20 วัน >20%")

    # --- Dynamic Fundamental Scoring (คิดคะแนนเฉพาะตัวที่มีข้อมูล) ---
    if d["eps_growth"] is not None:
        max_possible += 10
        if d["eps_growth"] > 0.10: 
            raw += 10
            reasons.append(f"EPS Growth เด่น ({d['eps_growth']*100:.1f}%)")
        elif d["eps_growth"] > 0: 
            raw += 5

    if d["revenue_growth"] is not None:
        max_possible += 10
        if d["revenue_growth"] > 0.10: 
            raw += 10
            reasons.append(f"Revenue Growth เด่น ({d['revenue_growth']*100:.1f}%)")
        elif d["revenue_growth"] > 0: 
            raw += 5

    if d["forward_pe"] is not None and d["pe"] is not None:
        max_possible += 5
        if d["forward_pe"] < d["pe"]: 
            raw += 5
            reasons.append("Forward P/E ถูกกว่า Trailing P/E")

    # Calculate normalized percentage score
    normalized_score = round((raw / max_possible) * 100) if max_possible > 0 else 0
    
    if normalized_score >= BUY_SCORE:
        label = "BUY CANDIDATE"
    elif normalized_score >= WATCH_SCORE:
        label = "WATCH"
    else:
        label = "AVOID / WAIT"

    return normalized_score, label, reasons, risk_flags


# ==========================================
# 6. EVENT DETECTION LOGIC
# ==========================================
def detect_events(d, label, last_state):
    """
    ตรวจจับ Event ตามลำดับความสำคัญ (Priority):
    1. BREAKOUT / BREAKDOWN (สำคัญที่สุด)
    2. NEAR_SUPPORT / NEAR_RESISTANCE
    3. NEW_BUY_SIGNAL
    """
    price = d["price"]
    prev_price = d["prev_price"]
    pivots = d["pivots"]
    s1, r1 = pivots["s1"], pivots["r1"]
    
    last_event = last_state.get("last_event")
    event_type = None
    event_desc = None

    # Priority 1: Breakout & Breakdown
    if prev_price <= r1 and price > r1:
        event_type = "BREAKOUT"
        event_desc = f"ผ่านแนวต้าน R1 ${r1:.2f}"
    elif prev_price >= s1 and price < s1:
        event_type = "BREAKDOWN"
        event_desc = f"หลุดแนวรับ S1 ${s1:.2f}"

    # Priority 2: Near Support & Resistance (ทำต่อเมื่อไม่มี Breakout/Breakdown)
    if not event_type:
        dist_s1 = abs(price - s1) / price
        dist_r1 = abs(price - r1) / price
        
        if dist_s1 <= LEVEL_NEAR_PCT:
            event_type = "NEAR_SUPPORT"
            event_desc = f"เข้าใกล้แนวรับ S1 (${s1:.2f})"
        elif dist_r1 <= LEVEL_NEAR_PCT:
            event_type = "NEAR_RESISTANCE"
            event_desc = f"เข้าใกล้แนวต้าน R1 (${r1:.2f})"

    # Priority 3: Status Upgraded to BUY CANDIDATE
    if not event_type:
        old_label = last_state.get("label")
        if label == "BUY CANDIDATE" and old_label != "BUY CANDIDATE":
            event_type = "BUY_CANDIDATE"
            event_desc = "สถานะอัปเกรดเป็น BUY CANDIDATE"

    # ป้องกันการส่งซ้ำ: ถ้าเป็น Event เดียวกับครั้งก่อน จะไม่ส่ง
    if event_type and event_type != last_event:
        return event_type, event_desc

    return None, None


# ==========================================
# 7. TELEGRAM MESSAGE FORMATTER
# ==========================================
def make_v2_message(ticker, d, score_val, label, event_type, event_desc):
    emoji = "🟢" if label == "BUY CANDIDATE" else ("🟡" if label == "WATCH" else "🔴")
    p = d["pivots"]
    macd_str = "Bullish 📈" if d["macd"] > d["macd_signal"] else "Bearish 📉"
    vol_ratio = (d['volume'] / d['vol20']) if d['vol20'] > 0 else 0.0

    # Header section
    msg = f"📢 <b>STOCK ALERT</b>\n<b>{event_type}</b>\n\n"
    if event_desc:
        msg += f"📍 {event_desc}\n\n"

    # Body section
    msg += (
        f"{emoji} <b>{ticker}</b> — <b>{label}</b>\n\n"
        f"⭐ <b>Score:</b> {score_val}/100\n"
        f"💰 <b>ราคา:</b> ${d['price']:.2f}\n\n"
        f"🟢 <b>แนวรับ</b>\n"
        f"• S1: ${p['s1']:.2f}\n"
        f"• S2: ${p['s2']:.2f}\n"
        f"• MA200: ${d['ma200']:.2f}\n\n"
        f"🔴 <b>แนวต้าน</b>\n"
        f"• R1: ${p['r1']:.2f}\n"
        f"• R2: ${p['r2']:.2f}\n"
        f"• 52W High: ${d['high52']:.2f}\n\n"
        f"📊 <b>RSI:</b> {d['rsi']:.1f}\n"
        f"📈 <b>MACD:</b> {macd_str}\n"
        f"📦 <b>Volume/Avg20:</b> {vol_ratio:.2f}x"
    )
    return msg


# ==========================================
# 8. MAIN EXECUTION LOOP
# ==========================================
def main():
    state = load_state()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting checks for {len(WATCHLIST)} stocks...")

    alert_count = 0
    for idx, ticker in enumerate(WATCHLIST, 1):
        print(f"[{idx}/{len(WATCHLIST)}] Checking {ticker}...", end="\r")
        d = fetch(ticker)
        if not d:
            continue

        s, label, reasons, risks = score(d)
        last_ticker_state = state.get(ticker, {})

        # Check for Events
        event_type, event_desc = detect_events(d, label, last_ticker_state)

        if event_type:
            msg = make_v2_message(ticker, d, s, label, event_type, event_desc)
            send_telegram(msg)
            last_ticker_state["last_event"] = event_type
            alert_count += 1
            time.sleep(1)  # Rate limiting delay for Telegram API

        # Update persistent state
        last_ticker_state.update({
            "label": label,
            "score": s,
            "last_price": d["price"],
            "updated": datetime.now(timezone.utc).isoformat()
        })
        state[ticker] = last_ticker_state

    save_state(state)
    print(f"\nExecution completed. Sent {alert_count} alerts.")

if __name__ == "__main__":
    main()
