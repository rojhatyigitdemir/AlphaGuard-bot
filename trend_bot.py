import sys
import os
import logging

# Log gürültülerini tamamen sustur
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)

f = open(os.devnull, 'w')
sys.stderr = f

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import time
import datetime as dt
from zoneinfo import ZoneInfo
import requests
import html
import re
from google import genai
from google.genai import types

warnings.filterwarnings('ignore')

# ====================================================================
# --- SETTINGS (AYARLAR) ---
# ====================================================================
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

client = genai.Client(api_key=API_KEY) if API_KEY else None
MODEL_ID = 'gemini-2.5-flash'

CORE_ASSETS = ["O", "BNDW", "BTC-USD", "ZGLD.SW", "SHEL", "ALL", "MU", "KLAC", "ZSIL.SW", "BCHE.SW"]

# SADECE BU LİSTEDEKİ VARLIKLAR ETF (FON) OLARAK KABUL EDİLECEKTİR (Uzantısız Kök Semboller)
ETF_LIST = [
    "ZSIL", "ZGLD", "CHDVD", "BCHE", "IWMO", "INFR", 
    "MVSH", "MSE", "VUSD", "XDPU", "SBIO", "XDGU", 
    "VDEA", "VAGX", "BNDW", "VOO", "PBE", "PLATC", "PALLC", "VSML"
]

DATA_PERIOD = "2y"
DATA_INTERVAL = "1wk"
SMA_WINDOW = 8
MOMENTUM_WEEKS = 4
MOMENTUM_COL = f"{MOMENTUM_WEEKS}W Mom (%)"
RETURN_1W_COL = "1W Ret (%)"

REBALANCE_WEEKDAYS = {0, 4} # Pazartesi (0) ve Cuma (4)

HISTORY_FILE = "signals_history.csv"
HISTORY_COLUMNS = [
    "run_date", "symbol", "category", "price", "trend",
    f"momentum_{MOMENTUM_WEEKS}w", "ai_signal",
    "eval_date_1w", "realized_return_1w",
    "eval_date_4w", "realized_return_4w",
    "opt_pcr", "opt_iv_spread", "opt_put_skew", "opt_flag",
]

# --- OPSİYON (PUT) SİNYALİ ---
# Cremers & Weinbaum (2010): ATM call IV - put IV farkı (IV spread) haftalık getiriyi öngörür;
#   put'ların göreli pahalı olması (negatif spread) bilgili satış/korunma baskısına işaret eder.
# Xing, Zhang & Zhao (2010): OTM put IV - ATM call IV (put skew) yüksekse sonraki getiriler zayıftır.
# Her iki sinyal de kesitsel (cross-sectional) sıralama ile kullanılır, mutlak eşik ile değil.
OPT_MIN_DAYS, OPT_MAX_DAYS = 7, 60
OPT_FLAG_QUANTILE = 0.20
OPT_MIN_VALID = 10

# Analist konsensüsü eşikleri (oy payı)
ANALYST_BUY_SHARE = 0.60
ANALYST_SELL_SHARE = 0.25

# Haftalık getiri hedefi (%). Değerlendirme için referans; risk kısıtı değildir.
WEEKLY_TARGET_PCT = 1.0
EVAL_DAYS_1W = 7
EVAL_DAYS_4W = 28


def get_swiss_now():
    try:
        return dt.datetime.now(ZoneInfo("Europe/Zurich"))
    except Exception:
        return dt.datetime.now()

def read_portfolio(file_name="portfolio.csv"):
    try:
        df = pd.read_csv(file_name)
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].dropna().tolist()
        else:
            symbols = df.iloc[:, 0].dropna().tolist()
        return [str(s).strip().upper() for s in symbols if str(s).strip()]
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return []

def secure_ai_query(prompt, max_retries=4):
    if not client:
        return "API Key Eksik."
    
    delays = [5, 15, 30, 45]
    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192
            )
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=config
            )
            if response and response.text:
                return response.text.strip()
            return "Boş yanıt"
        except Exception as e:
            wait_time = delays[attempt] if attempt < len(delays) else 30
            print(f"   [Gemini API Uyarısı] Deneme {attempt+1}/{max_retries} başarısız. {wait_time}s bekleniyor...")
            time.sleep(wait_time)
            
    return "AI Sunucu Hatası / Kota Aşımı"

def global_macro_intelligence():
    macro_symbols = ["^GSPC", "CL=F", "^TNX", "BTC-USD"]
    macro_text = ""
    for ms in macro_symbols:
        try:
            ticker = yf.Ticker(ms)
            news = ticker.news
            if news:
                for h in news[:2]:
                    title = h.get('title') or (h.get('content', {}).get('title') if isinstance(h.get('content'), dict) else None)
                    if title:
                        macro_text += f"- {title}\n"
        except Exception:
            pass

    if not macro_text:
        return "Calm trend in global markets."
    
    prompt = f"You are a chief economist. Read these headlines:\n{macro_text}\nWrite a sharp, 60-word GLOBAL STRATEGIC TACTICAL NOTE in Turkish. Plain prose, no headings, no markdown."
    return secure_ai_query(prompt).replace('\n', ' ')

def defcon_shock_monitor(symbols, macro_note):
    print("\n🔍 DEFCON Protokolü Başlatıldı...")
    alerts = []
    news_dataset = ""
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period="1mo", interval="1d")
            if hist.empty or len(hist) < 15:
                continue
            hist['TR'] = hist[['High', 'Low', 'Close']].max(axis=1) - hist[['High', 'Low', 'Close']].min(axis=1)
            yesterday_atr = hist['TR'].rolling(window=14).mean().iloc[-2]
            today_high, today_low, prev_close = hist['High'].iloc[-1], hist['Low'].iloc[-1], hist['Close'].iloc[-2]
            
            if pd.isna(yesterday_atr) or yesterday_atr == 0:
                continue
            gap_up = today_high - prev_close
            gap_down = prev_close - today_low
            
            up_pct = f"{(gap_up / prev_close) * 100:.1f}".replace(".", ",")
            down_pct = f"{(gap_down / prev_close) * 100:.1f}".replace(".", ",")
            if gap_up > (2.5 * yesterday_atr):
                alerts.append(f"📈 {sym} +{up_pct}% · fiyat sıçraması (2,5× ATR)")
            elif gap_down > (2.5 * yesterday_atr):
                alerts.append(f"📉 {sym} −{down_pct}% · sert düşüş (2,5× ATR)")

            news = yf.Ticker(sym).news
            titles = [h.get('title') for h in news[:2]] if news else []
            if titles:
                news_dataset += f"- {sym}: {' | '.join(titles)}\n"
        except Exception:
            continue

    if news_dataset:
        prompt = f"""
Makro Durum: {macro_note}
Haberler:
{news_dataset}

Sadece oyun değiştirici haberlere ŞOK de (diğerlerini yazma):
Format (Her varlık için bir satır):
SEMBOL | 🚀 YÜKSELİŞ ŞOKU | Gerekçe
SEMBOL | 🚨 DÜŞÜŞ ŞOKU | Gerekçe
"""
        raw_response = secure_ai_query(prompt)
        for line in raw_response.split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    sym = parts[0].strip().replace('*', '')
                    kind, reason = parts[1].strip(), parts[2].strip()
                    if "YÜKSELİŞ ŞOKU" in kind:
                        alerts.append(f"🧠 {sym} · yükseliş haberi — {reason}")
                    elif "DÜŞÜŞ ŞOKU" in kind:
                        alerts.append(f"🧠 {sym} · düşüş haberi — {reason}")
    return alerts

def is_rebalance_day():
    return get_swiss_now().weekday() in REBALANCE_WEEKDAYS

def _pick_option_expiry(expiries):
    """İlk vadeyi (çoğu zaman 0-2 günlük, spekülatif) değil, 7-60 gün aralığındaki en yakın vadeyi seçer."""
    today = get_swiss_now().date()
    fallback = None
    for e in expiries:
        try:
            days = (dt.date.fromisoformat(e) - today).days
        except Exception:
            continue
        if OPT_MIN_DAYS <= days <= OPT_MAX_DAYS:
            return e
        if days > OPT_MAX_DAYS and fallback is None:
            fallback = e
    return fallback

def _clean_option_leg(df):
    """Bozuk IV ve açık pozisyonu olmayan kontratları ayıklar; kotasyonu olanları tercih eder."""
    iv = df['impliedVolatility']
    valid = iv.between(0.01, 3.0) & (df['openInterest'].fillna(0) > 0)
    quoted = valid & (df['bid'].fillna(0) > 0) & (df['ask'].fillna(0) > 0)
    return df[quoted] if quoted.sum() >= 3 else df[valid]

def get_option_put_signals(t_obj, spot):
    """PCR (bağlam), IV spread (Cremers-Weinbaum) ve put skew (Xing-Zhang-Zhao) hesaplar. Değerler % puan."""
    out = {"pcr": np.nan, "iv_spread": np.nan, "put_skew": np.nan}
    expiries = t_obj.options
    if not expiries:
        return out
    exp = _pick_option_expiry(expiries)
    if exp is None:
        return out
    chain = t_obj.option_chain(exp)
    calls, puts = chain.calls, chain.puts

    c_vol, p_vol = calls['volume'].sum(), puts['volume'].sum()
    if c_vol > 0:
        out["pcr"] = p_vol / c_vol

    if not spot or spot <= 0:
        return out
    c, p = _clean_option_leg(calls), _clean_option_leg(puts)

    # IV spread: aynı strike'ta call IV - put IV, ATM bölgesi (K/S 0.9-1.1), açık pozisyon ağırlıklı
    m = c[['strike', 'impliedVolatility', 'openInterest']].merge(
        p[['strike', 'impliedVolatility', 'openInterest']], on='strike', suffixes=('_c', '_p'))
    m = m[(m['strike'] / spot).between(0.9, 1.1)]
    if len(m) >= 2:
        w = m['openInterest_c'].fillna(0) + m['openInterest_p'].fillna(0)
        if w.sum() > 0:
            out["iv_spread"] = float(np.average(m['impliedVolatility_c'] - m['impliedVolatility_p'], weights=w)) * 100

    # Put skew: K/S 0.95'e en yakın OTM put IV - K/S 1.0'a en yakın ATM call IV
    otm_put = p[(p['strike'] / spot).between(0.80, 0.95)]
    atm_call = c[(c['strike'] / spot).between(0.95, 1.05)]
    if not otm_put.empty and not atm_call.empty:
        put_iv = otm_put.loc[((otm_put['strike'] / spot) - 0.95).abs().idxmin(), 'impliedVolatility']
        call_iv = atm_call.loc[((atm_call['strike'] / spot) - 1.0).abs().idxmin(), 'impliedVolatility']
        out["put_skew"] = float(put_iv - call_iv) * 100
    return out

def assign_option_flags(df):
    """Evren içinde kesitsel sıralama: en pahalı put'lu %20 -> OptRisk:High, en ucuz -> OptRisk:Low."""
    df["Opt_Flag"] = ""
    ivs, skew = df.get("Opt_IVSpread"), df.get("Opt_PutSkew")
    if ivs is None or ivs.notna().sum() < OPT_MIN_VALID:
        return df
    ivs_lo, ivs_hi = ivs.quantile(OPT_FLAG_QUANTILE), ivs.quantile(1 - OPT_FLAG_QUANTILE)
    skew_hi = skew.quantile(1 - OPT_FLAG_QUANTILE) if skew.notna().sum() >= OPT_MIN_VALID else np.inf
    for idx in df.index:
        v, s = ivs.at[idx], skew.at[idx]
        if pd.isna(v):
            continue
        if v <= ivs_lo or (pd.notna(s) and s >= skew_hi):
            flag = "OptRisk:High"
        elif v >= ivs_hi:
            flag = "OptRisk:Low"
        else:
            flag = "OptRisk:Neutral"
        df.at[idx, "Opt_Flag"] = flag
        df.at[idx, "AltData (Smart Money)"] = str(df.at[idx, "AltData (Smart Money)"]).rstrip("]") + f" | {flag}]"
    return df

def parse_analyst_consensus(rec):
    """
    yfinance 'recommendations' tablosunu (period, strongBuy, buy, hold, sell, strongSell) okur.
    Eski sürüm kolon adlarına bakıyordu; bu kolonlar her zaman var olduğundan sonuç hep 'Ana:AL' çıkıyordu.
    Analist tavsiyeleri sistematik olarak iyimser olduğu için SAT eşiği AL eşiğinden düşük tutulur.
    """
    if rec is None or rec.empty:
        return "N/A"
    needed = ["strongBuy", "buy", "hold", "sell", "strongSell"]
    if not all(c in rec.columns for c in needed):
        return "N/A"
    row = rec[rec["period"] == "0m"].iloc[0] if "period" in rec.columns and (rec["period"] == "0m").any() else rec.iloc[0]
    counts = row[needed].astype(float).fillna(0)
    total = counts.sum()
    if total < 3:
        return "N/A"
    buy_share = (counts["strongBuy"] + counts["buy"]) / total
    sell_share = (counts["sell"] + counts["strongSell"]) / total
    if sell_share >= ANALYST_SELL_SHARE:
        label = "SAT"
    elif buy_share >= ANALYST_BUY_SHARE:
        label = "AL"
    else:
        label = "TUT"
    return f"Ana:{label}({buy_share * 100:.0f}% al, n={int(total)})"

def get_smart_money_data(symbol, spot=None):
    """(AltData metni, opsiyon sayısal değerleri) döndürür."""
    symbol_upper = symbol.upper()
    empty_opt = {"pcr": np.nan, "iv_spread": np.nan, "put_skew": np.nan}

    # Kural 1: Kripto Paralar
    if "-USD" in symbol_upper:
        try:
            req = requests.get("https://api.alternative.me/fng/?limit=1", timeout=3)
            if req.status_code == 200:
                data = req.json()['data'][0]
                return f"[Kripto F&G: {data['value']} ({data['value_classification']})]", empty_opt
        except Exception:
            pass
        return "[Kripto AltData Yok]", empty_opt

    # --- ZEKİ UZANTI TEMİZLEYİCİ ---
    base_symbol = symbol_upper.split('.')[0]

    # Kural 2: Manuel ETF Beyaz Listesi (Whitelist)
    if base_symbol in ETF_LIST:
        return "[ETF / Fon - İçsel Veri Yok]", empty_opt

    # Kural 3: Geriye Kalan Her Şey Gerçek Hisse Senedidir (ABD + Avrupa Şirketleri)
    else:
        opt_str, ins_str, ana_str = "N/A", "N/A", "N/A"
        opt = dict(empty_opt)
        try:
            t_obj = yf.Ticker(symbol)

            # Opsiyon Zinciri: PCR + IV spread + put skew
            try:
                opt = get_option_put_signals(t_obj, spot)
                parts = []
                if pd.notna(opt["pcr"]): parts.append(f"PCR:{opt['pcr']:.2f}")
                if pd.notna(opt["iv_spread"]): parts.append(f"IVS:{opt['iv_spread']:+.1f}")
                if pd.notna(opt["put_skew"]): parts.append(f"Skew:{opt['put_skew']:.1f}")
                if parts: opt_str = " ".join(parts)
            except Exception:
                pass
            
            # İçeriden Öğrenenler (Insider)
            try:
                idf = t_obj.insider_transactions
                if idf is not None and not idf.empty:
                    txt = idf.head(8).to_string().lower()
                    sales = txt.count('sale') + txt.count('sell')
                    buys = txt.count('purchase') + txt.count('buy')
                    if sales > buys: ins_str = "Ins:Satış"
                    elif buys > sales: ins_str = "Ins:Alış"
                    else: ins_str = "Ins:Nötr"
            except Exception:
                pass
            
            # Analist Tavsiyeleri
            try:
                ana_str = parse_analyst_consensus(t_obj.recommendations)
            except Exception:
                pass
            
            return f"[{opt_str} | {ins_str} | {ana_str}]", opt
        except Exception:
            return "[AltData Hatası]", opt

def batch_download_data(symbols, period=DATA_PERIOD, interval=DATA_INTERVAL):
    try:
        data = yf.download(tickers=symbols, period=period, interval=interval, group_by='ticker', threads=True, progress=False)
        return None if data is None or data.empty else data
    except Exception:
        return None

def analyze_asset_data(symbol, batch_data=None):
    try:
        data = None
        if batch_data is not None:
            if isinstance(batch_data.columns, pd.MultiIndex):
                if symbol in batch_data.columns.get_level_values(0):
                    data = batch_data[symbol].dropna(how='all')
            else:
                data = batch_data.dropna(how='all')

        if data is None:
            data = yf.download(symbol, period=DATA_PERIOD, interval=DATA_INTERVAL, progress=False)
        if data.empty or len(data) < SMA_WINDOW + MOMENTUM_WEEKS + 3:
            return None

        close_series = (data['Close'][symbol] if isinstance(data.columns, pd.MultiIndex) else data['Close']).ffill()
        volume_series = (data['Volume'][symbol] if isinstance(data.columns, pd.MultiIndex) else data['Volume']).ffill()

        if close_series.isna().all():
            return None

        last_bar_date = pd.Timestamp(close_series.index[-1]).tz_localize(None)
        today = pd.Timestamp(dt.date.today())
        current_week_open = today - pd.Timedelta(days=today.weekday())
        is_last_bar_open_week = last_bar_date >= current_week_open

        current_price = float(close_series.iloc[-1])
        
        if is_last_bar_open_week and len(close_series) > 1:
            closed_close = close_series.iloc[:-1]
            closed_volume = volume_series.iloc[:-1]
        else:
            closed_close = close_series
            closed_volume = volume_series

        if len(closed_close) < SMA_WINDOW + MOMENTUM_WEEKS + 1:
            return None

        sma_ref = closed_close.rolling(window=SMA_WINDOW).mean().dropna().iloc[-1]
        trend = "UPTREND 🟢" if current_price > sma_ref else "DOWNTREND 🔴"

        momentum_ref_price = closed_close.iloc[-(MOMENTUM_WEEKS)] 
        momentum_pct = ((current_price - momentum_ref_price) / momentum_ref_price) * 100
        
        one_week_ref_price = closed_close.iloc[-1]
        one_week_pct = ((current_price - one_week_ref_price) / one_week_ref_price) * 100

        volume_change = 0
        try:
            last_full_week_vol = closed_volume.iloc[-1]
            prev_avg_vol = closed_volume.iloc[-(MOMENTUM_WEEKS + 1):-1].mean()
            if prev_avg_vol > 0:
                volume_change = ((last_full_week_vol - prev_avg_vol) / prev_avg_vol) * 100
            if volume_change > 15: volume_comment = "Increasing (Strong)"
            elif volume_change < -15: volume_comment = "Decreasing (Weak)"
            else: volume_comment = "Stable"
        except Exception:
            volume_comment = "No Data"

        alt_data_str, opt = get_smart_money_data(symbol, spot=current_price)

        return {
            "Opt_PCR": opt["pcr"],
            "Opt_IVSpread": opt["iv_spread"],
            "Opt_PutSkew": opt["put_skew"],
            "Asset": symbol,
            "Price ($)": round(current_price, 2),
            "Absolute Trend": trend,
            RETURN_1W_COL: round(one_week_pct, 2),
            MOMENTUM_COL: round(momentum_pct, 2),
            "Volume_Num": volume_change, 
            "Volume Status": volume_comment,
            "AltData (Smart Money)": alt_data_str,
            "AI Action & Risk Warning": "---"
        }
    except Exception:
        return None

def dual_momentum_and_risk_analysis(symbols, macro_note):
    results = []
    sys.stdout = sys.__stdout__
    print(f"AlphaGuard AI Initiating Rebalance...\nStage 1: Calculating Data & AltData...\n")

    all_symbols = list(dict.fromkeys(list(symbols) + CORE_ASSETS))
    batch_data = batch_download_data(all_symbols)

    for symbol in symbols:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            results.append(asset_data)

    df = pd.DataFrame(results)

    if not df.empty:
        uptrend_assets = df[df["Absolute Trend"] == "UPTREND 🟢"].copy().sort_values(by=MOMENTUM_COL, ascending=False)
        if not uptrend_assets.empty:
            uptrend_assets['Category'] = 'Dynamic Uptrend'

        downtrend_assets = df[df["Absolute Trend"] == "DOWNTREND 🔴"].copy().sort_values(by=RETURN_1W_COL, ascending=False)
        if not downtrend_assets.empty:
            downtrend_assets['Category'] = 'Downtrend / Reversal'
        
        all_portfolio_assets = pd.concat([uptrend_assets, downtrend_assets], ignore_index=True)
    else:
        all_portfolio_assets = pd.DataFrame()

    core_results = []
    for symbol in CORE_ASSETS:
        asset_data = analyze_asset_data(symbol, batch_data=batch_data)
        if asset_data:
            asset_data['Category'] = 'Core Foundation'
            core_results.append(asset_data)
        else:
            core_results.append({
                "Asset": symbol, "Price ($)": 0.0, "Absolute Trend": "UNKNOWN", RETURN_1W_COL: 0.0,
                MOMENTUM_COL: 0.0, "Volume_Num": 0.0, "Volume Status": "No Data", "AltData (Smart Money)": "N/A",
                "AI Action & Risk Warning": "Data Error", "Category": 'Core Foundation'
            })

    final_analysis_list = pd.concat([pd.DataFrame(core_results), all_portfolio_assets], ignore_index=True)
    final_analysis_list = assign_option_flags(final_analysis_list)

    print(f"\nStage 2: AI Analysis (Bulletproof Plain Text Parser, Batch: 15)...\n")
    analysis_dict = {}
    BATCH_SIZE = 15
    
    time.sleep(5)
    
    for i in range(0, len(final_analysis_list), BATCH_SIZE):
        chunk = final_analysis_list.iloc[i:i+BATCH_SIZE]
        batch_serialized_data = ""
        for index, row in chunk.iterrows():
            symbol = row["Asset"]
            try:
                sys.stdout = f
                ticker = yf.Ticker(symbol)
                news = ticker.news
                sys.stdout = sys.__stdout__
                
                news_titles = [h.get('title') or "" for h in news[:2]] if news else []
                news_text = " | ".join([t for t in news_titles if t]) if news_titles else "No news."
            except Exception: 
                sys.stdout = sys.__stdout__
                news_text = "No news."
            
            batch_serialized_data += f"- Asset: {symbol}, Trend: {row['Absolute Trend']}, Ret: {row[RETURN_1W_COL]}%, Vol: {row['Volume Status']}, AltData: {row['AltData (Smart Money)']}, News: {news_text}\n"

        batch_prompt = f"""
        You are a hedge fund manager. Macro: {macro_note}
        Analyze these assets:
        {batch_serialized_data}

        INSTRUCTIONS:
        1. Evaluate EVERY asset. Use ONE tag: [STRONG BUY 🚀], [ACCUMULATE 🟢], [HOLD 🟡], [TRIM 🟠], or [SELL 🔴].
        2. Max 10-word justification, written in Turkish, plain text (no markdown).
        3. CRITICAL: Downgrade one level if AltData shows 'Ins:Satış' (Insider Selling) or 'OptRisk:High' (puts priced expensively vs calls = informed hedging). Upgrade if 'Ins:Alış'. 'Ana:SAT' is a meaningful negative (analysts rarely say sell); 'Ana:AL' alone is weak evidence. 'OptRisk:Low' is mildly positive. Treat PCR as context only.
        
        FORMAT (MUST FOLLOW THIS EXACTLY, ONE LINE PER ASSET):
        SEMBOL | [TAG] | Justification
        """

        raw_response = secure_ai_query(batch_prompt)
        
        try:
            for line in raw_response.split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        sym = parts[0].strip().replace('*', '') 
                        tag = parts[1].strip()
                        justification = parts[2].strip()
                        analysis_dict[sym] = f"{tag} {justification}"
        except Exception as e:
            print(f"Batch {i//BATCH_SIZE + 1} Parse Error: {e}")
        
        time.sleep(10)

    for index, row in final_analysis_list.iterrows():
        final_analysis_list.at[index, "AI Action & Risk Warning"] = analysis_dict.get(row["Asset"], "Hold and monitor (Rate Limit/AI Issue).")

    df_display = final_analysis_list.drop(columns=["Volume_Num", "Volume Status"])
    return df_display

def load_signal_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            for col in ["run_date", "eval_date_1w", "eval_date_4w"]:
                if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
            for col in HISTORY_COLUMNS:
                if col not in df.columns: df[col] = pd.NA
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=HISTORY_COLUMNS)

def get_latest_price(symbol):
    try:
        data = yf.Ticker(symbol).history(period="5d")
        if not data.empty:
            return float(data['Close'].ffill().iloc[-1])
    except Exception:
        pass
    return None

def update_realized_returns(history_df):
    if history_df.empty:
        return history_df
    today = pd.Timestamp(dt.date.today())
    price_cache = {}

    for idx, row in history_df.iterrows():
        run_date = row.get("run_date")
        if pd.isna(run_date): continue
        days_passed = (today - run_date).days
        entry_price = row.get("price")

        needs_1w = days_passed >= EVAL_DAYS_1W and pd.isna(row.get("realized_return_1w"))
        needs_4w = days_passed >= EVAL_DAYS_4W and pd.isna(row.get("realized_return_4w"))
        if not (needs_1w or needs_4w): continue

        if row["symbol"] not in price_cache:
            price_cache[row["symbol"]] = get_latest_price(row["symbol"])
        current_price = price_cache[row["symbol"]]
        
        if current_price is None or pd.isna(entry_price) or float(entry_price) == 0.0:
            continue

        realized_return = round(((current_price - entry_price) / entry_price) * 100, 2)
        if needs_1w:
            history_df.at[idx, "realized_return_1w"] = realized_return
            history_df.at[idx, "eval_date_1w"] = today
        if needs_4w:
            history_df.at[idx, "realized_return_4w"] = realized_return
            history_df.at[idx, "eval_date_4w"] = today
    return history_df

def _round_or_na(value, digits):
    try:
        return pd.NA if value is None or pd.isna(value) else round(float(value), digits)
    except Exception:
        return pd.NA

def append_new_signals(history_df, final_analysis_list):
    today = pd.Timestamp(dt.date.today())
    new_rows = []
    for _, row in final_analysis_list.iterrows():
        new_rows.append({
            "run_date": today, "symbol": row["Asset"], "category": row["Category"],
            "price": row["Price ($)"], "trend": row["Absolute Trend"], f"momentum_{MOMENTUM_WEEKS}w": row[MOMENTUM_COL],
            "ai_signal": row["AI Action & Risk Warning"], "eval_date_1w": pd.NaT,
            "realized_return_1w": pd.NA, "eval_date_4w": pd.NaT, "realized_return_4w": pd.NA,
            "opt_pcr": _round_or_na(row.get("Opt_PCR"), 3),
            "opt_iv_spread": _round_or_na(row.get("Opt_IVSpread"), 2),
            "opt_put_skew": _round_or_na(row.get("Opt_PutSkew"), 2),
            "opt_flag": row.get("Opt_Flag") or pd.NA,
        })
    return pd.concat([history_df, pd.DataFrame(new_rows)], ignore_index=True)

def generate_accuracy_summary(history_df):
    def is_hit(row, col):
        ret = row.get(col)
        if pd.isna(ret): return None
        if any(tag in str(row.get("ai_signal", "")) for tag in ["SELL", "TRIM"]): return ret < 0
        return ret > 0

    if "realized_return_1w" not in history_df.columns: return "Veri hazir degil."
    evaluated_1w = history_df.dropna(subset=["realized_return_1w"]).copy()
    if evaluated_1w.empty: return "Henuz 1 haftasi dolmus sinyal yok."

    evaluated_1w["hit"] = evaluated_1w.apply(lambda r: is_hit(r, "realized_return_1w"), axis=1)
    summary = f"[1 Hafta] Sinyal: {len(evaluated_1w)} | Ort. Getiri: {evaluated_1w['realized_return_1w'].mean():.2f}% | Isabet: {evaluated_1w['hit'].mean() * 100:.1f}%"
    mean_1w = evaluated_1w['realized_return_1w'].mean()
    summary += f"\n[Hedef] Haftalik %{WEEKLY_TARGET_PCT:.1f} | Fark: {mean_1w - WEEKLY_TARGET_PCT:+.2f} puan"

    if "realized_return_4w" in history_df.columns:
        evaluated_4w = history_df.dropna(subset=["realized_return_4w"]).copy()
        if not evaluated_4w.empty:
            evaluated_4w["hit"] = evaluated_4w.apply(lambda r: is_hit(r, "realized_return_4w"), axis=1)
            summary += f"\n[4 Hafta] Sinyal: {len(evaluated_4w)} | Ort. Getiri: {evaluated_4w['realized_return_4w'].mean():.2f}% | Isabet: {evaluated_4w['hit'].mean() * 100:.1f}%"
    return summary

# ====================================================================
# --- TELEGRAM ARAYÜZÜ ---
# Tasarım ilkesi: basit ama etkili. Önce özet, sonra detay; her varlık en fazla iki satır.
# ====================================================================
TG_LIMIT = 3800
TG_GROUP_DETAIL_MAX = 8
TG_REASON_MAX = 80
DAYS_TR = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
MONTHS_TR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
SIGNALS = [  # (model etiketi, simge, Türkçe ad)
    ("STRONG BUY", "🚀", "Güçlü al"),
    ("ACCUMULATE", "🟢", "Biriktir"),
    ("HOLD", "🟡", "Tut"),
    ("TRIM", "🟠", "Azalt"),
    ("SELL", "🔴", "Sat"),
]
NO_SIGNAL = (None, "⚪", "Sinyal yok")
DIVIDER = "━━━━━━━━━━━━━━━"

def _esc(text):
    return html.escape(str(text), quote=False)

def _num(x, digits=1, sign=False):
    """Türkçe ondalık biçimi: 8,2 / +8,2 / −3,1"""
    if x is None or pd.isna(x):
        return "–"
    txt = f"{abs(float(x)):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if sign:
        return ("+" if float(x) >= 0 else "−") + txt
    return ("−" if float(x) < 0 else "") + txt

def _header(title):
    now = get_swiss_now()
    stamp = f"{DAYS_TR[now.weekday()]} {now.day} {MONTHS_TR[now.month - 1]} · {now.strftime('%H:%M')}"
    return f"🛡️ <b>AlphaGuard</b> · {title}\n<i>{stamp} CH</i>"

def parse_signal(text):
    """'[STRONG BUY 🚀] gerekçe' -> ((etiket, simge, ad), gerekçe)"""
    text = str(text or "")
    upper = text.upper()
    for sig in SIGNALS:
        if sig[0] in upper:
            reason = text.split("]", 1)[1] if "]" in text else re.sub(sig[0], "", text, flags=re.I)
            reason = reason.strip(" -–|:")
            if len(reason) > TG_REASON_MAX:
                reason = reason[:TG_REASON_MAX - 1].rstrip() + "…"
            return sig, reason
    return NO_SIGNAL, ""

def _asset_lines(row):
    sig, reason = parse_signal(row.get("AI Action & Risk Warning"))
    trend = str(row.get("Absolute Trend", ""))
    arrow = "▲" if "UP" in trend else ("▼" if "DOWN" in trend else "•")
    mom = row.get(MOMENTUM_COL)
    badges = []
    if row.get("Opt_Flag") == "OptRisk:High":
        badges.append("⚠️ put pahalı")
    if "Ins:Satış" in str(row.get("AltData (Smart Money)", "")):
        badges.append("👤 insider satış")
    line1 = f"{sig[1]} <b>{_esc(row['Asset'])}</b>  {_num(row.get('Price ($)'), 2)}  {arrow} {_num(mom, 1, sign=True)}%"
    if badges:
        line1 += "  " + " · ".join(badges)
    line2 = f"      <i>{_esc(reason)}</i>" if reason else ""
    return [line1] + ([line2] if line2 else [])

def _signal_counts_line(df):
    counts = {sig[0]: 0 for sig in SIGNALS}
    for txt in df["AI Action & Risk Warning"]:
        sig, _ = parse_signal(txt)
        if sig[0]:
            counts[sig[0]] += 1
    return "  ".join(f"{sig[1]} {counts[sig[0]]}" for sig in SIGNALS)

def _shock_block(shock_alerts):
    if not shock_alerts:
        return "⚡ <b>Şok izleme</b>\n✅ Fiyat kırılımı veya kritik haber yok."
    lines = ["⚡ <b>Şok izleme</b>"] + [f"• {_esc(a)}" for a in shock_alerts[:12]]
    if len(shock_alerts) > 12:
        lines.append(f"<i>+{len(shock_alerts) - 12} uyarı daha</i>")
    return "\n".join(lines)

def build_daily_message(macro_note, shock_alerts):
    return "\n\n".join([
        _header("Günlük izleme"),
        f"🌍 <b>Piyasa notu</b>\n{_esc(macro_note)}",
        _shock_block(shock_alerts),
    ])

def compute_scorecard(history_df):
    """Tekrarlı kayıtları ayıklayıp 1 haftalık sinyal karnesini çıkarır."""
    if history_df is None or history_df.empty or "realized_return_1w" not in history_df.columns:
        return None
    df = history_df.dropna(subset=["realized_return_1w"]).copy()
    if df.empty:
        return None
    df = df.drop_duplicates(subset=["run_date", "symbol"], keep="last")
    df["ret"] = pd.to_numeric(df["realized_return_1w"], errors="coerce")
    df = df.dropna(subset=["ret"])
    if df.empty:
        return None
    df["sig"] = df["ai_signal"].map(lambda t: parse_signal(t)[0][0])
    bearish = df["sig"].isin(["SELL", "TRIM"])
    bullish = df["sig"].isin(["STRONG BUY", "ACCUMULATE"])
    hit = np.where(bearish, df["ret"] < 0, df["ret"] > 0)
    spread = df.loc[bullish, "ret"].mean() - df.loc[bearish, "ret"].mean() if bullish.any() and bearish.any() else np.nan
    return {
        "n": len(df),
        "mean": df["ret"].mean(),
        "hit": hit.mean() * 100,
        "base": (df["ret"] > 0).mean() * 100,
        "spread": spread,
    }

def build_scorecard_message(history_df):
    sc = compute_scorecard(history_df)
    title = "📊 <b>Sinyal karnesi</b> · 1 haftalık"
    if sc is None:
        return f"{title}\nHenüz değerlendirilecek sinyal yok."
    gap = sc["mean"] - WEEKLY_TARGET_PCT
    rows = [
        ("Ortalama getiri", f"{_num(sc['mean'], 2, sign=True)}%"),
        ("Haftalık hedef", f"{_num(WEEKLY_TARGET_PCT, 2, sign=True)}%"),
        ("Hedeften fark", f"{_num(gap, 2, sign=True)} puan"),
        ("İsabet oranı", f"%{_num(sc['hit'])}"),
        ("Piyasa tabanı", f"%{_num(sc['base'])}"),
        ("Al − Sat farkı", f"{_num(sc['spread'], 2, sign=True)} puan"),
        ("Örneklem", f"{sc['n']:,}".replace(",", ".") + " sinyal"),
    ]
    width = max(len(k) for k, _ in rows)
    table = "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)
    status = "✅ Hedefin üzerinde" if gap >= 0 else "⏳ Hedefin altında"
    note = "<i>Piyasa tabanı: tüm sinyallerde pozitif getiri payı. İsabet bunu geçmiyorsa sinyal katkısı yok.</i>"
    return f"{title}\n<pre>{_esc(table)}</pre>\n{status}\n{note}"

def _section(title, rows_df, sort_col=None, ascending=False, detail_max=None):
    if rows_df.empty:
        return []
    if sort_col and sort_col in rows_df.columns:
        rows_df = rows_df.sort_values(sort_col, ascending=ascending)
    lines = [title]
    shown = rows_df if detail_max is None else rows_df.head(detail_max)
    for _, row in shown.iterrows():
        lines += _asset_lines(row)
    rest = rows_df.iloc[len(shown):]
    if not rest.empty:
        lines.append(f"<i>+{len(rest)} diğer: {_esc(', '.join(rest['Asset'].astype(str)))}</i>")
    return lines

def _chunk_lines(lines, header):
    """Satırları bölmeden Telegram sınırına göre mesajlara ayırır."""
    messages, current = [], header
    for line in lines:
        if len(current) + len(line) + 1 > TG_LIMIT:
            messages.append(current)
            current = header + " <i>(devam)</i>"
        current += "\n" + line
    messages.append(current)
    return messages

def build_full_report_messages(macro_note, final_report_df, history_df, shock_alerts):
    df = final_report_df.copy()
    df["_sig"] = df["AI Action & Risk Warning"].map(lambda t: parse_signal(t)[0][0])
    core = df[df["Category"] == "Core Foundation"]
    sats = df[df["Category"] != "Core Foundation"]

    # 1) Özet: piyasa notu, şoklar, sinyal dağılımı
    summary = "\n\n".join([
        _header("Rebalance raporu"),
        f"🧭 <b>Bugünün sinyalleri</b>\n{_signal_counts_line(df)}\n<i>Fiyat · ▲▼ trend · 4 haftalık momentum</i>",
        f"🌍 <b>Piyasa notu</b>\n{_esc(macro_note)}",
        _shock_block(shock_alerts),
    ])
    messages = [summary]

    # 2) Çekirdek portföy
    core_lines = _section("", core, sort_col=MOMENTUM_COL)[1:]
    messages += _chunk_lines(core_lines, f"🏛️ <b>Çekirdek portföy</b> · {len(core)} varlık\n{DIVIDER}")

    # 3) Uydu varlıklar: önce aksiyon gerektirenler, "Tut" en sonda kısa liste
    sat_lines = []
    for code, icon, name in SIGNALS:
        group = sats[sats["_sig"] == code]
        if group.empty:
            continue
        title = f"\n{icon} <b>{name}</b> ({len(group)})"
        if code == "HOLD":
            sat_lines += [title, f"<i>{_esc(', '.join(group['Asset'].astype(str)))}</i>"]
        else:
            bearish = code in ("TRIM", "SELL")
            sat_lines += _section(title, group, sort_col=MOMENTUM_COL, ascending=bearish, detail_max=TG_GROUP_DETAIL_MAX)
    missing = sats[sats["_sig"].isna()]
    if not missing.empty:
        sat_lines += [f"\n⚪ <b>Sinyal yok</b> ({len(missing)})", f"<i>{_esc(', '.join(missing['Asset'].astype(str)))}</i>"]
    messages += _chunk_lines(sat_lines, f"🛰️ <b>Uydu varlıklar</b> · {len(sats)} varlık\n{DIVIDER}")

    # 4) Karne
    messages.append(build_scorecard_message(history_df))
    return messages

def _strip_html(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text))

def send_telegram_message(messages):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: 
        print("❌ HATA: Token veya Chat ID boş!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    if isinstance(messages, str):
        messages = [messages]
    for msg in messages:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                # HTML ayrıştırılamazsa mesaj kaybolmasın: düz metin olarak tekrar dene
                print(f"Telegram API Hatası: {resp.status_code} - {resp.text}")
                requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": _strip_html(msg)}, timeout=15)
        except Exception as e:
            print(f"Telegram Gönderim Hatası: {e}")
        time.sleep(1)

if __name__ == "__main__":
    try:
        print("\n🚀 ALPHAGUARD SİSTEMİ BAŞLATILIYOR...")
        watchlist = read_portfolio("portfolio.csv")
        watchlist = [s for s in watchlist if s not in CORE_ASSETS] if watchlist else []
        
        all_monitored_symbols = list(set(CORE_ASSETS + watchlist))

        # 1. MAKRO & DEFCON ŞOK İZLEMESİ
        macro_note = global_macro_intelligence()
        shock_alerts = defcon_shock_monitor(all_monitored_symbols, macro_note)

        # 2. REBALANCE
        if is_rebalance_day():
            final_report = dual_momentum_and_risk_analysis(watchlist, macro_note)
            pd.set_option('display.max_colwidth', None)
            
            history_df = update_realized_returns(load_signal_history())
            history_df = append_new_signals(history_df, final_report)
            cols = [c for c in history_df.columns if c in HISTORY_COLUMNS or "1m" in c or "3m" in c]
            history_df[cols].to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
            
            print(generate_accuracy_summary(history_df))
            report_messages = build_full_report_messages(macro_note, final_report, history_df, shock_alerts)
            print("\n\n".join(report_messages))
            send_telegram_message(report_messages)
        else:
            report_text = build_daily_message(macro_note, shock_alerts)
            print(report_text)
            send_telegram_message(report_text)
            
        print("\n🏁 SİSTEM BAŞARIYLA TAMAMLANDI!")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR (Sistem Çöktü): {e}")
        raise e
