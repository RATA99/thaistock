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
    prev_close  = df['close'].iloc[-2]
    chg_pct     = (current_p - prev_close) / prev_close * 100
    avg_vol     = df['volume'].tail(20).mean()
    last_vol    = df['volume'].iloc[-1]
    vol_ratio   = last_vol / avg_vol if avg_vol > 0 else 1
    last10      = df.tail(10)[['time','open','high','low','close','volume']].to_dict(orient='records')
    swing_high  = df['high'].tail(50).max()
    swing_low   = df['low'].tail(50).min()
    last_high   = df['high'].iloc[-1]
    last_low    = df['low'].iloc[-1]

    closes      = df['close'].tail(50).round(0)
    support_levels = closes.value_counts().head(3).index.tolist()
    ema_signal  = "Golden Cross (Bullish)" if ema50 > ema200 else "Death Cross (Bearish)"
    vol_signal  = "Volume Spike — รอยเท้าเจ้ามือ (Big Lot)" if vol_ratio >= 2 else f"Volume ปกติ ({vol_ratio:.1f}x avg)"

    return f"""🛡️ The Alpha Sniper Protocol (v.2026 High-Win-Rate)

Role: คุณคือ Senior Quant Strategist & Chief Investment Officer (CIO) ประสบการณ์ 30 ปี ผู้เชี่ยวชาญ Confluence Analysis เพื่อหาจุดเข้าเทรดที่มี Win Rate สูงสุด

══════════════════════════════════════
[INPUT DATA SECTION]
══════════════════════════════════════
- Symbol          : {symbol}
- Current Price   : {current_p:,.2f} บาท ({chg_pct:+.2f}% จากแท่งก่อน)
- Timeframe       : {interval_label}
- Sector/Industry : SET — วิเคราะห์จากข้อมูลที่มี
- Market Sentiment: ประเมินจากข้อมูลด้านล่าง

Macro Data:
- High ล่าสุด  : {last_high:,.2f} | Low ล่าสุด: {last_low:,.2f}
- Swing High   : {swing_high:,.2f} | Swing Low (50 แท่ง): {swing_low:,.2f}
- EMA 50       : {ema50:,.2f} | EMA 200: {ema200:,.2f} → {ema_signal}
- Fibonacci    : 38.2% = {fibo['38.2%']:,.2f} | 50% = {fibo['50%']:,.2f} | 61.8% = {fibo['61.8%']:,.2f}
- แนวรับ Statistical (top 3): {', '.join([f'{x:,.0f}' for x in support_levels])}

Micro Data:
- Volume ล่าสุด: {last_vol:,.0f} หุ้น → {vol_signal}
- 10 แท่งล่าสุด: {json.dumps(last10, default=str, ensure_ascii=False)}

══════════════════════════════════════
[STEP-BY-STEP ANALYSIS PROTOCOL]
══════════════════════════════════════
วิเคราะห์ครบทุกขั้นตอน ภาษาไทย กระชับ เน้นตัวเลขและหลักฐาน:

**1. 🏭 Sector Correlation Check**
วิเคราะห์ว่า {symbol} เคลื่อนไหวสอดคล้องกับ Sector หรือเป็น Outperformer เทียบตลาด?

**2. 📰 Fundamental & Catalyst**
ประเมินปัจจัยพื้นฐานและ sentiment จากข้อมูล price action ที่มี — มีสัญญาณ Smart Money สะสมหรือระบายหรือไม่?

**3. 🔭 Macro Structure**
- EMA Trend: {ema_signal} — ความหมายต่อแนวโน้มหลัก
- Fibonacci Confluence: ราคาอยู่โซนใด และโซนถัดไปคืออะไร
- Major Support/Resistance จาก Statistical levels

**4. ⚡ Micro Trigger**
- VPA: {vol_signal} — อ่าน "รอยเท้าเจ้ามือ" จาก Volume + Price
- Price Pattern ที่เห็นใน 10 แท่งล่าสุด (Double Bottom / Bull Flag / H&S ฯลฯ)
- Candlestick Signal จุดกลับตัว (Engulfing / Hammer / Pin Bar)

**5. 🚫 Zero Trust Validation**
ถ้าสัญญาณ Macro + Micro ขัดแย้งกัน หรือ R:R < 1:2 → สั่ง **SKIP** ทันที พร้อมระบุเหตุผล

══════════════════════════════════════
[OUTPUT: THE SNIPER BLUEPRINT]
══════════════════════════════════════

**🎯 Strategic Verdict: [BUY / WAIT / SKIP]**
เหตุผล 3 ข้อที่คมที่สุด:
1. ...
2. ...
3. ...

**📋 Execution Table:**
| | ราคา | เหตุผล |
|---|---|---|
| Entry Point | ... | ... |
| Take Profit 1 | ... | ... |
| Take Profit 2 | ... | ... |
| Stop Loss | ... | ... |

**📐 Risk/Reward Ratio:** คำนวณให้เห็นตัวเลขชัดเจน (ต้องได้ ≥ 1:2)

**👁️ Insider Monitoring:** สิ่งที่ต้องจับตาดูใน 24 ชั่วโมงข้างหน้า"""

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
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        return res.choices[0].message.content
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
        st.subheader("🤖 บทวิเคราะห์ (Powered by Groq × Llama 3.3 70B)")

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

        if analyze_clicked and not st.session_state.get('analyzing', False):
            st.session_state.analyzing = True
            with st.spinner(f"🧠 กำลังวิเคราะห์ {symbol} ({selected_label})..."):
                st.session_state.analysis_text  = get_ai_analysis(
                    symbol, selected_label, df.copy(), fibo, current_p, ema50, ema200
                )
                st.session_state.analysis_label = (
                    f"{symbol} • {selected_label} • {datetime.now(BKK).strftime('%H:%M:%S')} (ICT)"
                )
            st.session_state.analyzing = False

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
