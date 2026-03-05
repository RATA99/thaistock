import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from settrade_v2 import Investor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from openai import OpenAI
import time
import json

from config import APP_ID, APP_SECRET, BROKER_ID, APP_CODE, GROQ_API_KEY

BKK = ZoneInfo("Asia/Bangkok")

st.set_page_config(page_title="SET Stock Sniper", layout="wide")

# ─── CONNECTIONS ──────────────────────────────────────────────────────────────
@st.cache_resource
def init_settrade():
    return Investor(
        app_id=APP_ID,
        app_secret=APP_SECRET,
        broker_id=BROKER_ID,
        app_code=APP_CODE,
        is_auto_queue=False
    )

# ─── FETCH DATA ────────────────────────────────────────────────────────────────
def get_data(symbol: str, interval: str = "1d", limit: int = 200):
    investor = init_settrade()
    market   = investor.MarketData()
    res      = market.get_candlestick(
        symbol=symbol.upper().strip(),
        interval=interval,
        limit=limit
    )
    if not res:
        return pd.DataFrame()

    data = {k: v for k, v in res.items() if isinstance(v, list)}
    if not data or 'time' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.columns = [c.lower() for c in df.columns]

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.sort_values('time').reset_index(drop=True)
    return df

# ─── INDICATORS ───────────────────────────────────────────────────────────────
def calculate_indicators(df):
    if df.empty or len(df) < 2:
        return None, 0, 0

    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA50']  = df['close'].ewm(span=50,  adjust=False).mean()

    high_p = df['high'].max()
    low_p  = df['low'].min()
    diff   = high_p - low_p

    fibo = {
        '38.2%': high_p - 0.382 * diff,
        '50%':   high_p - 0.500 * diff,
        '61.8%': high_p - 0.618 * diff,
    }
    return fibo, high_p, low_p

# ─── ZOOM RANGE ───────────────────────────────────────────────────────────────
def get_xaxis_range(df, interval):
    if df.empty:
        return None, None
    latest = df['time'].iloc[-1]
    zoom_window = {
        "1m":  timedelta(hours=2),
        "5m":  timedelta(hours=6),
        "15m": timedelta(hours=12),
        "30m": timedelta(days=3),
        "60m": timedelta(days=5),
    }
    if interval in zoom_window:
        x_start = max(latest - zoom_window[interval], df['time'].iloc[0])
        return x_start, latest + timedelta(minutes=30)
    return df['time'].iloc[0], latest + timedelta(days=1)

# ─── AI ANALYSIS ──────────────────────────────────────────────────────────────
def build_prompt(symbol, interval_label, df, fibo, current_p, ema50, ema200):
    prev_close   = df['close'].iloc[-2]
    chg_pct      = (current_p - prev_close) / prev_close * 100
    avg_vol      = df['volume'].tail(20).mean()
    last_vol     = df['volume'].iloc[-1]
    vol_ratio    = last_vol / avg_vol if avg_vol > 0 else 1
    last10       = df.tail(10)[['time','open','high','low','close','volume']].to_dict(orient='records')

    # ✅ ใช้ High/Low 30 แท่งล่าสุด
    recent_high  = df['high'].tail(30).max()
    recent_low   = df['low'].tail(30).min()
    diff         = recent_high - recent_low

    # ✅ สูตรที่ถูกต้อง: Fibo = Low + (High - Low) × Level
    fib_236 = recent_low + diff * 0.236
    fib_382 = recent_low + diff * 0.382
    fib_500 = recent_low + diff * 0.500
    fib_618 = recent_low + diff * 0.618
    fib_786 = recent_low + diff * 0.786

    last_high  = df['high'].iloc[-1]
    last_low   = df['low'].iloc[-1]
    ema_signal = "เหนือ EMA 200 ✅ Bullish Zone" if current_p > ema200 else "ใต้ EMA 200 ❌ Bearish Zone"

    # ✅ Volume interpretation ที่ถูกต้อง
    if vol_ratio >= 3.0:
        vol_signal = f"🔴 Volume Spike รุนแรง {vol_ratio:.1f}x — Big Player เข้า/ออก ต้องดูทิศทางราคาด้วย"
    elif vol_ratio >= 2.0:
        vol_signal = f"🟠 Volume Spike {vol_ratio:.1f}x — มีแรงผิดปกติ ดูว่าราคาขึ้นหรือลงพร้อมกัน"
    elif vol_ratio >= 1.3:
        vol_signal = f"🟡 Volume สูงเล็กน้อย {vol_ratio:.1f}x — ยังไม่ชัดเจน รอยืนยัน"
    else:
        vol_signal = f"⚪ Volume เบาบาง {vol_ratio:.1f}x — LOW VOLUME REBOUND ⚠️ ไม่มีแรงซื้อจริง อย่าเชื่อ"

    # หาโซน Fibo ปัจจุบัน
    if current_p >= fib_786:
        fib_zone = f"เหนือ 78.6% ({fib_786:,.2f}) — ใกล้ High แล้ว Risk/Reward ต่ำ"
    elif current_p >= fib_618:
        fib_zone = f"61.8%–78.6% ({fib_618:,.2f}–{fib_786:,.2f}) — โซนแนวต้านแข็งแกร่ง"
    elif current_p >= fib_500:
        fib_zone = f"50%–61.8% ({fib_500:,.2f}–{fib_618:,.2f}) — ด่านมหาหินกลาง"
    elif current_p >= fib_382:
        fib_zone = f"38.2%–50% ({fib_382:,.2f}–{fib_500:,.2f}) — ยืนเหนือ 38.2% แล้ว เป้าถัดไป {fib_500:,.2f}"
    elif current_p >= fib_236:
        fib_zone = f"23.6%–38.2% ({fib_236:,.2f}–{fib_382:,.2f}) — ยังไม่เบรก 38.2% อย่าเพิ่งเข้า"
    else:
        fib_zone = f"ต่ำกว่า 23.6% ({fib_236:,.2f}) — โซนต่ำมาก ยังไม่ควรเข้า"

    return f"""Role: คุณคือ Senior Quant Trader ประสบการณ์ 30 ปี ผู้เชี่ยวชาญตลาดหุ้นไทย (SET)
เป้าหมาย: Win Rate สูงสุด ด้วยวินัยเหล็ก — ถ้า Setup ไม่สมบูรณ์ 100% ต้องกล้า SKIP

══════════════════════════════════════
⚠️ CRITICAL RULES (ละเมิดไม่ได้)
══════════════════════════════════════
1. ใช้เฉพาะข้อมูลใน [INPUT DATA] ห้ามใช้ตัวเลข Fibo หรือ EMA จากความทรงจำ
2. R:R < 1:2 → SKIP ทันที แม้สัญญาณจะดูดี ห้าม BUY เด็ดขาด
3. Volume เบาบาง (< 1.3x) = Low Volume Rebound ≠ Smart Money สะสม
4. คำนวณ Fibo ด้วยสูตร: Price = Low + (High − Low) × Level เท่านั้น

══════════════════════════════════════
[INPUT DATA — ใช้ตัวเลขเหล่านี้เท่านั้น]
══════════════════════════════════════
Symbol         : {symbol}
Timeframe      : {interval_label}
Current Price  : {current_p:,.2f} บาท ({chg_pct:+.2f}% จากแท่งก่อน)

[Macro]
30-Bar High    : {recent_high:,.2f}
30-Bar Low     : {recent_low:,.2f}
EMA 50         : {ema50:,.2f}
EMA 200        : {ema200:,.2f} → {ema_signal}
High/Low ล่าสุด: {last_high:,.2f} / {last_low:,.2f}

[Fibonacci — คำนวณด้วยสูตร Low + (High−Low)×Level]
23.6% = {fib_236:,.2f}
38.2% = {fib_382:,.2f}  ← ด่านแรก ต้องเบรกผ่านก่อนเข้า
50.0% = {fib_500:,.2f}  ← ด่านมหาหิน (Target 1)
61.8% = {fib_618:,.2f}  ← Golden Ratio (Target 2)
78.6% = {fib_786:,.2f}

📍 ราคา {current_p:,.2f} อยู่ที่: {fib_zone}

[Micro]
Volume         : {last_vol:,.0f} หุ้น → {vol_signal}
10 แท่งล่าสุด : {json.dumps(last10, default=str, ensure_ascii=False)}

══════════════════════════════════════
[ANALYSIS — ทำตามลำดับ อ้างอิงตัวเลขจริงทุกข้อ]
══════════════════════════════════════

**1. 📐 Fibonacci Zone Check**
ยืนยันโซน Fibo ของราคา {current_p:,.2f} และระบุ:
- แนวต้านถัดไป = ?
- แนวรับถัดไป = ?
(ใช้ตัวเลขจาก [INPUT DATA] เท่านั้น ห้ามประดิษฐ์)

**2. 📊 Volume Truth Test**
วิเคราะห์ "{vol_signal}" ให้ตรงตามความจริง:
- ถ้า Volume เบาบาง → "Low Volume Rebound ยังไม่มีแรงซื้อจริง อย่าหลอกตัวเอง"
- ถ้า Volume Spike → ดูทิศทางราคา ขึ้น+Spike=Bullish / ลง+Spike=Distribution

**3. 🔭 Multi-Timeframe Confluence**
- Macro (EMA 200 = {ema200:,.2f}): ราคาอยู่{ema_signal} — นัยต่อแนวโน้มหลัก?
- Micro (10 แท่งล่าสุด): Price Pattern ที่เห็นชัด? (Double Bottom / Lower High / Inside Bar)
- Confluence Result: สัญญาณ Macro + Micro ชี้ทิศทางเดียวกัน? ใช่/ไม่ใช่

**4. 🚫 R:R Validation (บังคับคำนวณก่อนตัดสินใจ)**
- Entry ที่คิดไว้ = ?
- Target 1 = ? (Fibo ?)
- Stop Loss = ? (อิง Low หรือ EMA)
- Upside = Target − Entry = ?
- Downside = Entry − Stop = ?
- R:R = Upside / Downside = ?
- ถ้า R:R < 2.0 → SKIP ห้าม BUY ไม่มีข้อยกเว้น

══════════════════════════════════════
[FINAL OUTPUT: SNIPER BLUEPRINT]
══════════════════════════════════════

**🎯 Verdict: [BUY / WAIT / SKIP]**
(อ้างอิง Fibo/EMA/Volume จริงทุกข้อ ห้ามเดา)
1. ...
2. ...
3. ...

**📋 Execution Table:**
| | ราคา | เหตุผล (Fibo/EMA Level) |
|---|---|---|
| Entry | ... | รอยืนยันเหนือ Fibo ...% = ... |
| Target 1 | {fib_500:,.2f} | Fibo 50% |
| Target 2 | {fib_618:,.2f} | Fibo 61.8% |
| Stop Loss | ... | หลุด Low/EMA = ... |

**📐 R:R Calculation (แสดงตัวเลขชัดเจน):**
Upside  = Target1 ({fib_500:,.2f}) − Entry = ...
Downside = Entry − Stop = ...
R:R = ... : 1
→ [ผ่าน ≥ 2.0 / ❌ SKIP < 2.0]

**⚠️ Expert's Warning (24 ชม.):**
ระบุ Price Level ที่เป็นจุดเปลี่ยนเกมชัดเจน — ถ้าหลุด ... = จบเกม / ถ้าเบรก ... = เปิดไป ..."""
def get_ai_analysis(symbol, interval_label, df, fibo, current_p, ema50, ema200):
    if not GROQ_API_KEY:
        return "⚠️ ไม่พบ GROQ_API_KEY — กรุณาใส่ key ใน `.streamlit/secrets.toml`\n\n```toml\nGROQ_API_KEY = 'gsk_...'\n```"
    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )
        prompt = build_prompt(symbol, interval_label, df, fibo, current_p, ema50, ema200)
        res = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        raw = res.choices[0].message.content

        # ── R:R Safety Net ────────────────────────────────────────────────────
        # คำนวณ R:R จากข้อมูลจริง แล้ว override ถ้า model ไม่ enforce
        import re as _re
        rr_match = _re.search(r'R[:/]R\s*[=:]\s*(\d+[.]?\d*)\s*[:/]\s*1', raw, _re.IGNORECASE)
        buy_match = _re.search(r'Verdict.*BUY', raw, _re.IGNORECASE)

        if buy_match and rr_match:
            rr_val = float(rr_match.group(1))
            if rr_val < 2.0:
                override_msg = f"""
---
🚫 **[SYSTEM OVERRIDE — R:R Validation Failed]**
Model สรุป BUY แต่ R:R = **{rr_val:.2f}:1 < 2.0** ซึ่งต่ำกว่าเกณฑ์ที่กำหนด

✅ **Verdict ถูกต้องคือ: SKIP**
เหตุผล: Risk/Reward ไม่คุ้มค่า — รอ Setup ที่ R:R ≥ 2.0 ก่อนเข้าเทรด
"""
                raw = raw + override_msg

        return raw
    except Exception as e:
        err = str(e)
        if "429" in err:
            return "⏳ Groq rate limit — รอสักครู่แล้วกดวิเคราะห์ใหม่ครับ"
        return f"⚠️ Error: {err}"

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    'symbol':         'DELTA',
    'interval':       '1d',
    'auto_refresh':   True,
    'analysis_text':  '',
    'analysis_label': '',
    'analyzing':      False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── INTERVAL CONFIG ──────────────────────────────────────────────────────────
interval_map = {
    "1 นาที":     "1m",
    "5 นาที":     "5m",
    "15 นาที":    "15m",
    "30 นาที":    "30m",
    "1 ชั่วโมง":  "60m",
    "รายวัน":     "1d",
    "รายสัปดาห์": "1w",
}
intraday_intervals = {"1m", "5m", "15m", "30m", "60m"}

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("📈 SET Stock Sniper")

col_search, col_interval, col_toggle, col_status = st.columns([2, 2, 1.5, 2])

with col_search:
    symbol_input = st.text_input(
        "search",
        value=st.session_state.symbol,
        placeholder="🔍 พิมพ์ชื่อหุ้น เช่น DELTA, PTT, AOT...",
        label_visibility="collapsed",
    )
    if symbol_input.strip():
        st.session_state.symbol = symbol_input.strip().upper()

with col_interval:
    selected_label = st.selectbox(
        "interval",
        list(interval_map.keys()),
        index=5,
        label_visibility="collapsed",
    )
    st.session_state.interval = interval_map[selected_label]

with col_toggle:
    st.session_state.auto_refresh = st.toggle("🔄 Auto", value=st.session_state.auto_refresh)

with col_status:
    st.caption(f"⏱ Last update: **{datetime.now(BKK).strftime('%H:%M:%S')}** (ICT)")

st.divider()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
symbol      = st.session_state.symbol
interval    = st.session_state.interval
is_intraday = interval in intraday_intervals

try:
    df = get_data(symbol, interval, limit=200)

    if not df.empty:
        fibo, r_high, r_low = calculate_indicators(df)
        current_p = df['close'].iloc[-1]
        ema200    = df['EMA200'].iloc[-1]
        ema50     = df['EMA50'].iloc[-1]
        prev_p    = df['close'].iloc[-2]
        chg       = current_p - prev_p
        chg_pct   = chg / prev_p * 100
        vol       = df['volume'].iloc[-1]

        # ── Metrics ──
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"💰 {symbol}", f"{current_p:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
        m2.metric("High",           f"{df['high'].iloc[-1]:,.2f}")
        m3.metric("Low",            f"{df['low'].iloc[-1]:,.2f}")
        m4.metric("EMA 50 / 200",   f"{ema50:,.2f} / {ema200:,.2f}")
        m5.metric("Volume",         f"{vol:,.0f}")

        # ── Chart ──
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name=symbol,
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['EMA200'], name="EMA 200",
            line=dict(color='#FF6B35', width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['EMA50'], name="EMA 50",
            line=dict(color='#29B6F6', width=1.2, dash='dot'),
        ))
        fig.add_trace(go.Bar(
            x=df['time'], y=df['volume'], name="Volume",
            marker_color='rgba(100,100,255,0.3)', yaxis='y2',
        ))

        fibo_colors = {'38.2%': '#00E676', '50%': '#FFC107', '61.8%': '#FF5252'}
        for k, v in fibo.items():
            fig.add_hline(
                y=v, line_dash="dot", line_color=fibo_colors[k], line_width=1,
                annotation_text=f"Fibo {k}  {v:,.2f}", annotation_position="right"
            )

        x_start, x_end = get_xaxis_range(df, interval)
        if is_intraday and x_start is not None:
            zoom_df     = df[df['time'] >= x_start]
            y_min       = (zoom_df['low'].min()  if not zoom_df.empty else df['low'].min())  * 0.998
            y_max       = (zoom_df['high'].max() if not zoom_df.empty else df['high'].max()) * 1.002
            yaxis_range = [y_min, y_max]
        else:
            yaxis_range = None

        fig.update_layout(
            template="plotly_dark", height=560,
            xaxis=dict(
                range=[x_start, x_end],
                rangeslider=dict(visible=False),
                rangeselector=dict(
                    buttons=[
                        dict(count=30,  label="30m", step="minute", stepmode="backward"),
                        dict(count=1,   label="1h",  step="hour",   stepmode="backward"),
                        dict(count=3,   label="3h",  step="hour",   stepmode="backward"),
                        dict(count=1,   label="1d",  step="day",    stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#1e1e2e", activecolor="#FF6B35",
                ) if is_intraday else dict(visible=False),
            ),
            yaxis=dict(title="ราคา (บาท)", side="left", range=yaxis_range),
            yaxis2=dict(
                overlaying='y', side='right', showgrid=False,
                title="Volume", range=[0, df['volume'].max() * 5]
            ),
            margin=dict(l=10, r=130, t=30, b=10),
            legend=dict(orientation="h", y=1.02),
        )

        if is_intraday:
            zoom_labels = {
                "1m": "2 ชม.", "5m": "6 ชม.", "15m": "12 ชม.",
                "30m": "3 วัน", "60m": "5 วัน"
            }
            st.caption(f"🔎 zoom: {zoom_labels.get(interval, '')} ล่าสุด")

        st.plotly_chart(fig, use_container_width=True)

        # ── Signal ──
        if current_p >= fibo['38.2%']:
            st.success(f"💹 **BULLISH** — {symbol} แรงซื้อยังคุมตลาด | เป้าหมาย {fibo['38.2%']:,.2f}")
        elif current_p < ema200:
            st.error(f"🚨 **CRITICAL** — หลุด EMA 200 ({ema200:,.2f}) | ระวังขาลง")
        elif current_p < fibo['61.8%']:
            st.error(f"📉 **BEARISH** — ต่ำกว่า Fibo 61.8% = {fibo['61.8%']:,.2f}")
        elif current_p < fibo['50%']:
            st.warning(f"⚖️ **NEUTRAL** — ทดสอบแนวรับ Fibo 50% = {fibo['50%']:,.2f}")
        else:
            st.info(f"🔍 **WATCH** — อยู่ในช่วง Fibo 50%–38.2%")

        st.divider()

        # ── AI ANALYSIS ───────────────────────────────────────────────────────
        st.subheader("🤖 บทวิเคราะห์ (Powered by Groq × DeepSeek R1)")

        col_btn, col_lbl = st.columns([2, 5])
        with col_btn:
            analyze_clicked = st.button(
                f"🔍 วิเคราะห์ {symbol}",
                type="primary",
                use_container_width=True,
            )
        with col_lbl:
            if st.session_state.analysis_label:
                st.caption(f"วิเคราะห์ล่าสุด: {st.session_state.analysis_label}")

        if analyze_clicked:
            # reset flag ทุกครั้งที่กดปุ่ม เพื่อป้องกัน flag ค้าง
            st.session_state.analyzing = False
            with st.spinner(f"🧠 กำลังวิเคราะห์ {symbol} ({selected_label})..."):
                st.session_state.analysis_text  = get_ai_analysis(
                    symbol, selected_label, df.copy(), fibo, current_p, ema50, ema200
                )
                st.session_state.analysis_label = (
                    f"{symbol} • {selected_label} • {datetime.now(BKK).strftime('%H:%M:%S')} (ICT)"
                )

        if st.session_state.analysis_text:
            with st.container(border=True):
                st.markdown(st.session_state.analysis_text)

        with st.expander("📋 ข้อมูลย้อนหลัง (50 แท่งล่าสุด)"):
            st.dataframe(
                df[['time','open','high','low','close','volume']].tail(50)[::-1],
                use_container_width=True
            )

    else:
        st.warning(f"⚠️ ไม่พบข้อมูลสำหรับ **{symbol}** — กรุณาตรวจสอบชื่อหุ้น")

except Exception as e:
    err = str(e)
    if "Interval" in err or "interval" in err.lower():
        st.error(f"⚠️ Interval **'{interval}'** ไม่รองรับ — ลองเปลี่ยนเป็น รายวัน หรือ รายสัปดาห์")
    else:
        st.error(f"⚠️ Error: {err}")
        st.exception(e)

# ─── AUTO REFRESH ─────────────────────────────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(1)
    st.rerun()
