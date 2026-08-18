# Stock Signal Bot

บอทวิเคราะห์หุ้นและแจ้ง Telegram โดยใช้ Technical + Fundamental signals

## สิ่งที่ระบบทำ

- MA20 / MA50 / MA200
- RSI
- MACD
- 52-week high
- Volume เทียบค่าเฉลี่ย 20 วัน
- EPS Growth และ Revenue Growth เมื่อ Yahoo Finance มีข้อมูล
- ให้คะแนน 0-100 (normalized ตามข้อมูลที่มีจริง — ดูหัวข้อ "การแก้ไข" ด้านล่าง)
- BUY CANDIDATE / WATCH / AVOID
- ป้องกัน Telegram แจ้ง BUY ซ้ำด้วย `alert_state.json`
- Backtest แบบ technical-only ครอบคลุมทั้ง watchlist พร้อมประเมิน transaction cost

## การแก้ไขจากเวอร์ชันเดิม

1. **RSI edge case** — เดิมถ้าหุ้นไม่มีวันแดงเลยในช่วงคำนวณ (`avg_loss == 0`) RSI จะออกมาเป็น `NaN` แล้วหุ้นตัวนั้นหลุดจากการให้คะแนนไปเงียบๆ ตอนนี้ set เป็น RSI 100 ตรงๆ
2. **คะแนนลำเอียงเมื่อ fundamentals หาย** — เดิม fundamentals มีสัดส่วนคงที่ 25/100 คะแนน ถ้า yfinance ไม่มีข้อมูล (พบบ่อย) หุ้นจะเสียคะแนนไปฟรีๆ ทำให้แทบไม่มีทางถึง `BUY_SCORE = 80` ตอนนี้คะแนนคำนวณจาก raw score หารด้วย max score ที่ "มีข้อมูลจริง" ของหุ้นตัวนั้น แล้วค่อย scale เป็น 0-100
3. **Backtest ไม่มีทาง trade ได้เลย** — เดิม `backtest.py` ไม่ได้รวม fundamentals เข้าคะแนนแบบ `stock_bot.py` ทำให้คะแนนสูงสุดที่เป็นไปได้คือ 65/100 แต่ threshold ตั้งไว้ที่ 80 จึงไม่มีทาง trade เกิดขึ้นเลย ตอนนี้ backtest ใช้ scoring แบบ technical-only ที่ normalize เป็น 0-100 ของตัวเอง (และตั้งใจไม่ใส่ fundamentals เพราะ `.info` ให้ข้อมูลปัจจุบันเท่านั้น ใส่เข้าไปในอดีตจะเกิด look-ahead bias)
4. **Backtest ครอบคลุมทั้ง watchlist** — เดิมทดสอบแค่ NVDA ตัวเดียว ตอนนี้ loop ทุก ticker ใน watchlist แล้วสรุปทั้งภาพรวมและรายตัว
5. **หัก transaction cost โดยประมาณ** — เดิมไม่มี ตอนนี้หักค่าคอมมิชชั่น+slippage โดยประมาณ (`TRANSACTION_COST_PCT`) ต่อขาเข้า-ออก เพื่อให้ผลตอบแทนใกล้เคียงความจริงมากขึ้น
6. **Retry + sanity check ตอนดึงข้อมูล** — เพิ่ม retry with backoff รอบ `yf.download` / `yf.Ticker().info` และเช็คว่าแท่งราคาล่าสุดไม่เก่าเกินไป (Yahoo endpoint หลุด/ดีเลย์บ่อย)

**ข้อควรระวังที่ยังคงอยู่:** `yf.Ticker().info` (P/E, EPS growth, Revenue growth) ยังเป็นข้อมูลที่ไม่เสถียรและอาจขาดหาย/คลาดเคลื่อนได้ ถ้าต้องการความแม่นยำของ fundamentals จริงจัง ควรพิจารณาเปลี่ยนไปคำนวณจาก financial statements เอง หรือใช้ data vendor อื่น เช่น Financial Modeling Prep / Alpha Vantage / Polygon.io

## ตั้งค่า Telegram

สร้าง GitHub Secrets:

- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`

ห้ามใส่ Token ลงใน source code

## รันในเครื่อง

```bash
pip install -r requirements.txt
python stock_bot.py
```

รัน backtest (ทดสอบทั้ง watchlist):

```bash
python backtest.py
```

## หมายเหตุสำคัญ

ระบบนี้เป็นเครื่องมือช่วยกรองสัญญาณเบื้องต้น ไม่ใช่คำแนะนำการลงทุนและไม่รับประกันผลตอบแทน ควรใช้ประกอบการตัดสินใจร่วมกับการวิเคราะห์อื่นๆ เสมอ
