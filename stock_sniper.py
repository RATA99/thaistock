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
# ❌ ลบ @st.cache_resource ออก — ทำให้ token หมดอายุแล้วไม่ recreate
def get_investor():
    return Investor(
        app_id=APP_ID,
        app_secret=APP_SECRET,
        broker_id=BROKER_ID,
        app_code=APP_CODE,
        is_auto_queue=False
    )

# ─── FETCH DATA ────────────────────────────────────────────────────────────────
def get_data(symbol: str, interval: str = "1d", limit: int = 200):
    from settrade_v2.errors import SettradeError

    MAX_RETRY = 3
    for attempt in range(MAX_RETRY):
        try:
            # สร้าง investor ใหม่ทุกครั้ง = token ใหม่เสมอ
            investor = get_investor()
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

        except Exception as e:
            err = str(e).lower()
            if "access token" in err or "expired" in err or "invalid" in err:
                if attempt < MAX_RETRY - 1:
                    time.sleep(1)
                    continue  # retry ด้วย investor ใหม่
            raise  # error อื่นๆ throw ทิ้ง

    return pd.DataFrame()

# ─── INDICATORS ───────────────────────────────────────────────────────────────
def calculate_indicators(df):
    if df.empty or len(df) < 2:
        return None, 0, 0

    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA50']  = df['close'].ewm(span=50,  adjust=False).mean()

    high_p = df['high'].tail(30).max()
    low_p  = df['low'].tail(30).min()
    diff   = high_p - low_p

    fibo = {
        '23.6%': low_p + 0.236 * diff,
        '38.2%': low_p + 0.382 * diff,
        '50%':   low_p + 0.500 * diff,
        '61.8%': low_p + 0.618 * diff,
        '78.6%': low_p + 0.786 * diff,
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

    recent_high  = df['high'].tail(30).max()
    recent_low   = df['low'].tail(30).min()
    diff         = recent_high - recent_low

    fib_236 = recent_low + diff * 0.236
    fib_382 = recent_low + diff * 0.382
    fib_500 = recent_low + diff * 0.500
    fib_618 = recent_low + diff * 0.618
    fib_786 = recent_low + diff * 0.786

    last_high  = df['high'].iloc[-1]
    last_low   = df['low'].iloc[-1]
    ema_signal = "เหนือ EMA 200 ✅ Bullish Zone" if current_p > ema200 else "ใต้ EMA 200 ❌ Bearish Zone"

    if vol_ratio >= 3.0:
        vol_signal = f"🔴 Volume Spike รุนแรง {vol_ratio:.1f}x — Big Player เข้า/ออก"
    elif vol_ratio >= 2.0:
        vol_signal = f"🟠 Volume Spike {vol_ratio:.1f}x — มีแรงผิดปกติ"
    elif vol_ratio >= 1.3:
        vol_signal = f"🟡 Volume สูงเล็กน้อย {vol_ratio:.1f}x — ยังไม่ชัดเจน"
    else:
        vol_signal = f"⚪ Volume เบาบาง {vol_ratio:.1f}x — LOW VOLUME REBOUND ⚠️"

    if current_p >= fib_786:
        fib_zone = f"เหนือ 78.6% ({fib_786:,.2f}) — ใกล้ High แล้ว Risk/Reward ต่ำ"
    elif current_p >= fib_618:
        fib_zone = f"61.8%–78.6% ({fib_618:,.2f}–{fib_786:,.2f}) — โซนแนวต้านแข็งแกร่ง"
    elif current_p >= fib_500:
        fib_zone = f"50%–61.8% ({fib_500:,.2f}–{fib_618:,.2f}) — ด่านมหาหินกลาง"
    elif current_p >= fib_382:
        fib_zone = f"38.2%–50% ({fib_382:,.2f}–{fib_500:,.2f}) — ยืนเหนือ 38.2% เป้าถัดไป {fib_500:,.2f}"
    elif current_p >= fib_236:
        fib_zone = f"23.6%–38.2% ({fib_236:,.2f}–{fib_382:,.2f}) — ยังไม่เบรก 38.2%"
    else:
        fib_zone = f"ต่ำกว่า 23.6% ({fib_236:,.2f}) — โซนต่ำมาก"

    return f"""Role: Senior Quant Trader 30 ปี ผู้เชี่ยวชาญตลาดหุ้นไทย (SET)
เป้าหมาย: Win Rate สูงสุด — ถ้า Setup ไม่สมบูรณ์ ต้องกล้า SKIP

⚠️ CRITICAL RULES:
1. ใช้เฉพาะตัวเลขใน [INPUT DATA]
2. R:R < 1:2 → SKIP ทันที ห้าม BUY
3. Volume เบาบาง (< 1.3x) ≠ Smart Money
4. Fibo = Low + (High−Low) × Level

[INPUT DATA]
Symbol: {symbol} | TF: {interval_label}
Price: {current_p:,.2f} ({chg_pct:+.2f}%)
EMA50: {ema50:,.2f} | EMA200: {ema200:,.2f} → {ema_signal}
30-Bar High: {recent_high:,.2f} | Low: {recent_low:,.2f}
Fibo: 23.6%={fib_236:,.2f} | 38.2%={fib_382:,.2f} | 50%={fib_500:,.2f} | 61.8%={fib_618:,.2f} | 78.6%={fib_786:,.2f}
Zone: {fib_zone}
Volume: {vol_signal}
10 แท่ง: {json.dumps(last10, default=str, ensure_ascii=False)}

[ANALYSIS]
1. Fibonacci Zone + แนวรับ/แนวต้านถัดไป
2. Volume Truth Test
3. Multi-TF Confluence (EMA + Pattern)
4. R:R Validation (คำนวณให้เห็น)

[OUTPUT]
Verdict: BUY / WAIT / SKIP
Entry | TP1 ({fib_500:,.2f}) | TP2 ({fib_618:,.2f}) | SL
R:R Calculation
Expert Warning 24h"""

def get_ai_analysis(symbol, interval_label, df, fibo, current_p, ema50, ema200):
    if not GROQ_API_KEY:
        return "⚠️ ไม่พบ GROQ_API_KEY"
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        prompt = build_prompt(symbol, interval_label, df, fibo, current_p, ema50, ema200)
        res = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        raw = res.choices[0].message.content

        import re as _re
        rr_match  = _re.search(r'R[:/]R\s*[=:]\s*(\d+[.]?\d*)\s*[:/]\s*1', raw, _re.IGNORECASE)
        buy_match = _re.search(r'Verdict.*BUY', raw, _re.IGNORECASE)
        if buy_match and rr_match:
            rr_val = float(rr_match.group(1))
            if rr_val < 2.0:
                raw += f"\n\n---\n🚫 **[SYSTEM OVERRIDE]** R:R = {rr_val:.2f}:1 < 2.0 → **Verdict: SKIP**"
        return raw
    except Exception as e:
        err = str(e)
        if "429" in err:
            return "⏳ Groq rate limit — รอสักครู่แล้วกดใหม่"
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
    symbol_input = st.text_input("search", value=st.session_state.symbol,
        placeholder="🔍 พิมพ์ชื่อหุ้น เช่น DELTA, PTT...",
        label_visibility="collapsed")
    if symbol_input.strip():
        st.session_state.symbol = symbol_input.strip().upper()

with col_interval:
    selected_label = st.selectbox("interval", list(interval_map.keys()),
        index=5, label_visibility="collapsed")
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
    with st.spinner(f"Loading {symbol}..."):
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

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"💰 {symbol}", f"{current_p:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
        m2.metric("High",         f"{df['high'].iloc[-1]:,.2f}")
        m3.metric("Low",          f"{df['low'].iloc[-1]:,.2f}")
        m4.metric("EMA 50/200",   f"{ema50:,.2f} / {ema200:,.2f}")
        m5.metric("Volume",       f"{vol:,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name=symbol,
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA200'], name="EMA 200",
            line=dict(color='#FF6B35', width=1.5)))
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA50'], name="EMA 50",
            line=dict(color='#29B6F6', width=1.2, dash='dot')))
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name="Volume",
            marker_color='rgba(100,100,255,0.3)', yaxis='y2'))

        fibo_colors = {
            '23.6%': '#80CBC4', '38.2%': '#00E676',
            '50%':   '#FFC107', '61.8%': '#FF5252', '78.6%': '#CE93D8'
        }
        for k, v in fibo.items():
            fig.add_hline(y=v, line_dash="dot", line_color=fibo_colors[k],
                line_width=1, annotation_text=f"Fibo {k}  {v:,.2f}",
                annotation_position="right")

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
            xaxis=dict(range=[x_start, x_end],
                rangeslider=dict(visible=False),
                rangeselector=dict(buttons=[
                    dict(count=30, label="30m", step="minute", stepmode="backward"),
                    dict(count=1,  label="1h",  step="hour",   stepmode="backward"),
                    dict(count=3,  label="3h",  step="hour",   stepmode="backward"),
                    dict(count=1,  label="1d",  step="day",    stepmode="backward"),
                    dict(step="all", label="All"),
                ], bgcolor="#1e1e2e", activecolor="#FF6B35") if is_intraday else dict(visible=False)),
            yaxis=dict(title="ราคา (บาท)", side="left", range=yaxis_range),
            yaxis2=dict(overlaying='y', side='right', showgrid=False,
                title="Volume", range=[0, df['volume'].max() * 5]),
            margin=dict(l=10, r=130, t=30, b=10),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        if current_p < ema200:
            st.error(f"🚨 **CRITICAL** — หลุด EMA 200 ({ema200:,.2f})")
        elif current_p < fibo['38.2%']:
            st.error(f"📉 **BEARISH** — ต่ำกว่า Fibo 38.2% ({fibo['38.2%']:,.2f})")
        elif current_p < fibo['50%']:
            st.warning(f"⚖️ **WATCH** — โซน 38.2%–50%")
        elif current_p < fibo['61.8%']:
            st.info(f"🔍 **NEUTRAL** — โซน 50%–61.8%")
        elif current_p < fibo['78.6%']:
            st.success(f"💹 **BULLISH** — ยืนเหนือ 61.8% ({fibo['61.8%']:,.2f})")
        else:
            st.success(f"🚀 **STRONG BULL** — ยืนเหนือ 78.6% | ระวัง R:R")

        st.divider()
        st.subheader("🤖 บทวิเคราะห์ (Groq × Qwen3 32B)")

        col_btn, col_lbl = st.columns([2, 5])
        with col_btn:
            analyze_clicked = st.button(f"🔍 วิเคราะห์ {symbol}",
                type="primary", use_container_width=True)
        with col_lbl:
            if st.session_state.analysis_label:
                st.caption(f"ล่าสุด: {st.session_state.analysis_label}")

        if analyze_clicked:
            st.session_state.analyzing = False
            with st.spinner(f"🧠 กำลังวิเคราะห์ {symbol}..."):
                st.session_state.analysis_text = get_ai_analysis(
                    symbol, selected_label, df.copy(), fibo, current_p, ema50, ema200)
                st.session_state.analysis_label = (
                    f"{symbol} • {selected_label} • {datetime.now(BKK).strftime('%H:%M:%S')} ICT")

        if st.session_state.analysis_text:
            with st.container(border=True):
                st.markdown(st.session_state.analysis_text)

        with st.expander("📋 ข้อมูลย้อนหลัง 50 แท่ง"):
            st.dataframe(df[['time','open','high','low','close','volume']].tail(50)[::-1],
                use_container_width=True)
    else:
        st.warning(f"⚠️ ไม่พบข้อมูล **{symbol}**")

except Exception as e:
    err = str(e)
    if "interval" in err.lower():
        st.error(f"⚠️ Interval '{interval}' ไม่รองรับ")
    elif "access token" in err.lower() or "expired" in err.lower():
        st.error("⚠️ Settrade token หมดอายุ — กำลัง retry อัตโนมัติ...")
        st.rerun()
    else:
        st.error(f"⚠️ {err}")
        st.exception(e)

# ─── AUTO REFRESH ─────────────────────────────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(1)
    st.rerun()
