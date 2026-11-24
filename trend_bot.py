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

import macro_layer as ml
import portfolio_layer as pl

warnings.filterwarnings('ignore')

# ====================================================================
# --- SETTINGS (AYARLAR) ---
# ====================================================================
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# thinking ayarı v1beta'da destekleniyor; desteklenmezse secure_ai_query ayarı kendisi kaldırır
client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(api_version='v1beta')) if API_KEY else None
MODEL_ID = 'gemini-2.5-flash'
BATCH_SIZE = 12

# Gerçekte tutulan portföy (portfolio_layer.PORTFOLIO). Eski core varlıkları uydu listesine taşındı.
CORE_ASSETS = pl.PORTFOLIO_SYMBOLS

# SADECE BU LİSTEDEKİ VARLIKLAR ETF (FON) OLARAK KABUL EDİLECEKTİR (Uzantısız Kök Semboller)
ETF_LIST = [
    "ZSIL", "ZGLD", "CHDVD", "BCHE", "IWMO", "INFR", 
    "MVSH", "MSE", "VUSD", "XDPU", "SBIO", "XDGU", 
    "VDEA", "VAGX", "BNDW", "VOO", "PBE", "PLATC", "PALLC", "VSML", "XSPS", "XDEF", "XSPX"
]

DATA_PERIOD = "2y"
DATA_INTERVAL = "1wk"
SMA_WINDOW = 8
MOMENTUM_WEEKS = 4
MOMENTUM_COL = f"{MOMENTUM_WEEKS}W Mom (%)"
RETURN_1W_COL = "1W Ret (%)"

REBALANCE_WEEKDAYS = {1, 3} # Salı (1) ve Perşembe (3)
AI_FALLBACK_LABEL = "NO SIGNAL (AI yanıtı alınamadı)"

HISTORY_FILE = "signals_history.csv"
HISTORY_COLUMNS = [
    "run_date", "symbol", "category", "price", "trend",
    f"momentum_{MOMENTUM_WEEKS}w", "ai_signal",
    "eval_date_1w", "realized_return_1w",
    "eval_date_4w", "realized_return_4w",
    "opt_pcr", "opt_iv_spread", "opt_put_skew", "opt_flag",
    "regime", "macro_impact", "macro_driver",
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

# Ücretsiz Gemini kotası dakikada ~10 istekle sınırlı. Çağrılar arasında en az
# MIN_CALL_INTERVAL saniye bırakılır; 429 hatasında API'nin önerdiği bekleme süresi
# kullanılır, günlük kota biterse yedek modele geçilir. Bu görev akıl yürütme
# gerektirmediği için "thinking" kapalıdır: düşünme token'ları çıktı limitini
# yiyip yanıtı kesiyordu, bu da varlıkların yanıtsız kalmasına yol açıyordu.
# Yedek modeller sırayla denenir; mevcut olmayan (404) model listeden çıkarılır.
FALLBACK_MODELS = ['gemini-flash-lite-latest', 'gemini-flash-latest']
MIN_CALL_INTERVAL = 7.0
AI_STATS = {"calls": 0, "quota_errors": 0, "fallback_switches": 0, "failed": 0}
_ai_state = {"last_call": 0.0, "model": MODEL_ID, "fallbacks": list(FALLBACK_MODELS), "plain_config": set()}

def _ai_config(model):
    kwargs = dict(temperature=0.1, max_output_tokens=8192)
    if model not in _ai_state["plain_config"]:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)

def _err_code(err):
    code = getattr(err, "code", None)
    if code is None:
        m = re.match(r"\s*(\d{3})\b", str(err))
        code = int(m.group(1)) if m else None
    return code

def _is_quota_error(err):
    text = str(err)
    return _err_code(err) == 429 or "RESOURCE_EXHAUSTED" in text

def _is_daily_quota(err):
    text = str(err)
    return "PerDay" in text or "per day" in text.lower()

def _suggested_delay(err):
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(err))
    return min(float(m.group(1)) + 2, 90) if m else None

def _switch_model(reason):
    if _ai_state["fallbacks"]:
        old, new = _ai_state["model"], _ai_state["fallbacks"].pop(0)
        _ai_state["model"] = new
        AI_STATS["fallback_switches"] += 1
        print(f"   [Gemini] {old}: {reason}. {new} modeline geçiliyor.")
        return True
    if _ai_state["model"] != MODEL_ID:
        print(f"   [Gemini] {_ai_state['model']}: {reason}. Yedek kalmadı, {MODEL_ID} modeline dönülüyor.")
        _ai_state["model"] = MODEL_ID
        return True
    return False

def secure_ai_query(prompt, max_retries=4):
    if not client:
        return "API Key Eksik."

    backoff = [5, 15, 30, 45]
    quota_backoff = [20, 40, 60, 60]
    attempt, fixes = 0, 0
    while attempt < max_retries and fixes < 6:
        wait = MIN_CALL_INTERVAL - (time.time() - _ai_state["last_call"])
        if wait > 0:
            time.sleep(wait)
        _ai_state["last_call"] = time.time()
        AI_STATS["calls"] += 1
        model = _ai_state["model"]
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=_ai_config(model))
            text = (response.text or "").strip() if response else ""
            if text:
                return text
            print(f"   [Gemini] Boş yanıt ({model}), deneme {attempt+1}/{max_retries}.")
            time.sleep(backoff[min(attempt, len(backoff) - 1)])
            attempt += 1
            continue
        except Exception as e:
            code, text = _err_code(e), str(e)
            # Yapılandırma ya da model sorunu: beklemeden düzeltip yeniden dene
            if code == 400 and model not in _ai_state["plain_config"]:
                _ai_state["plain_config"].add(model)
                print(f"   [Gemini] {model} thinking ayarını kabul etmedi; ayar bu model için kaldırıldı.")
                fixes += 1
                continue
            if code == 404 and model != MODEL_ID:
                fixes += 1
                if _switch_model("model bulunamadı"):
                    continue
            if _is_quota_error(e):
                AI_STATS["quota_errors"] += 1
                quota_id = re.search(r"quotaId['\"]?\s*[:=]\s*['\"]?([A-Za-z\-]+)", text)
                print(f"   [Gemini] 429 {model}: {quota_id.group(1) if quota_id else 'kota'}")
                if (_is_daily_quota(e) or attempt >= 1) and _switch_model("kota doldu"):
                    fixes += 1
                    continue
                wait_time = _suggested_delay(e) or quota_backoff[min(attempt, len(quota_backoff) - 1)]
                print(f"   [Gemini] Kota sınırı ({model}). {wait_time:.0f}s bekleniyor, deneme {attempt+1}/{max_retries}.")
            else:
                wait_time = backoff[min(attempt, len(backoff) - 1)]
                print(f"   [Gemini] Hata ({type(e).__name__}): {text[:120]}. {wait_time}s bekleniyor, deneme {attempt+1}/{max_retries}.")
            time.sleep(wait_time)
            attempt += 1

    AI_STATS["failed"] += 1
    return "AI Sunucu Hatası / Kota Aşımı"

def defcon_shock_monitor(symbols, macro_note, news_reader):
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

            titles = news_reader.titles(sym, max_items=3)
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
    if symbol_upper.endswith("=F"):
        return "[Emtia vadeli - İçsel Veri Yok]", empty_opt

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

def dual_momentum_and_risk_analysis(symbols, macro, news_reader):
    macro_note = macro["note"]
    regime = macro["regime"]["label"]
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
        if not asset_data and symbol in pl.SYMBOL_FALLBACKS:  # aynı ürünün başka borsadaki kotasyonu
            asset_data = analyze_asset_data(pl.SYMBOL_FALLBACKS[symbol])
            if asset_data:
                asset_data["Asset"] = symbol
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

    # Makro duyarlılık: varlık betaları x haber tabanlı faktör şok puanları
    betas = ml.estimate_betas(batch_data, final_analysis_list["Asset"].tolist(), macro["factor_changes"])
    impact = ml.macro_impact(betas, macro["scores"])
    final_analysis_list["Macro_Impact"] = final_analysis_list["Asset"].map(impact["impact"]) if not impact.empty else np.nan
    final_analysis_list["Macro_Driver"] = final_analysis_list["Asset"].map(impact["driver"]) if not impact.empty else ""
    final_analysis_list["Macro_Flag"] = final_analysis_list["Asset"].map(impact["flag"]) if not impact.empty else ""
    final_analysis_list["Regime"] = regime

    print(f"\nStage 2: AI Analysis (Bulletproof Plain Text Parser, Batch: {BATCH_SIZE})...\n")
    analysis_dict, conviction_dict = {}, {}

    def asset_line(row):
        symbol = row["Asset"]
        titles, source = news_reader.asset_news(symbol, ETF_LIST)
        if titles:
            via = f" (via {source})" if source and source != symbol else ""
            news_text = f"{via} " + " | ".join(titles)
        else:
            news_text = "No news."
        macro_val = row.get("Macro_Impact")
        macro_text = f"{macro_val:+.1f}% ({row.get('Macro_Driver') or '-'})" if pd.notna(macro_val) else "n/a"
        inverse = " (INVERSE ETF)" if symbol.upper().split('.')[0] in ml.INVERSE_ETFS else ""
        return f"- Asset: {symbol}{inverse}, MacroImpact: {macro_text}, Trend: {row['Absolute Trend']}, Ret: {row[RETURN_1W_COL]}%, Vol: {row['Volume Status']}, AltData: {row['AltData (Smart Money)']}, News: {news_text}\n"

    def build_prompt(batch_serialized_data):
        return f"""
        You are a hedge fund manager. Market regime: {regime}. Macro: {macro_note}
        MacroImpact = expected weekly return effect (%) of this week's global shocks, given the asset's measured sensitivity to rates, USD, yen, oil, chips/AI, VIX, equities and crypto. The driver factor is in brackets.
        Analyze these assets:
        {batch_serialized_data}

        INSTRUCTIONS:
        1. Evaluate EVERY asset. Use ONE tag: [STRONG BUY 🚀], [ACCUMULATE 🟢], [HOLD 🟡], [TRIM 🟠], or [SELL 🔴].
        2. Max 10-word justification, written in Turkish, plain text (no markdown).
        3. CRITICAL: Downgrade one level if AltData shows 'Ins:Satış' (Insider Selling) or 'OptRisk:High' (puts priced expensively vs calls = informed hedging). Upgrade if 'Ins:Alış'. 'Ana:SAT' is a meaningful negative (analysts rarely say sell); 'Ana:AL' alone is weak evidence. 'OptRisk:Low' is mildly positive. Treat PCR as context only.
        4. MACRO: Downgrade one level if MacroImpact <= -1%; it supports an upgrade only if the asset's own data agree. In a Risk-off regime do not use STRONG BUY. News marked 'via' comes from the ETF's underlying, not the fund itself.
        
        5. CONVICTION: give 1-5 for how strongly the evidence supports your tag (5 = clear, consistent evidence; 3 = mixed; 1 = guess). Be strict: transaction costs are high, so 4-5 only when trend, data, news and macro agree.
        6. Assets in category 'Core Foundation' are positions the investor already holds; judge whether to keep them.

        FORMAT (MUST FOLLOW THIS EXACTLY, ONE LINE PER ASSET):
        SEMBOL | [TAG] | Justification | CONVICTION
        """

    def parse_response(raw_response):
        for line in raw_response.split('\n'):
            if '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue
            sym = parts[0].strip().replace('*', '').strip('-• ').upper()
            analysis_dict[sym] = f"{parts[1].strip()} {parts[2].strip()}"
            if len(parts) >= 4:
                m = re.search(r"[1-5]", parts[3])
                if m:
                    conviction_dict[sym] = int(m.group(0))

    lines = {row["Asset"]: asset_line(row) for _, row in final_analysis_list.iterrows()}
    symbols = list(lines)
    for i in range(0, len(symbols), BATCH_SIZE):
        pending = symbols[i:i + BATCH_SIZE]
        for attempt in range(3):  # yanıtsız kalanlar daha küçük bir istekle tekrar sorulur
            try:
                parse_response(secure_ai_query(build_prompt("".join(lines[s] for s in pending))))
            except Exception as e:
                print(f"Batch {i//BATCH_SIZE + 1} Parse Error: {e}")
            pending = [s for s in pending if s.upper() not in analysis_dict]
            if not pending:
                break
            print(f"   Batch {i//BATCH_SIZE + 1}: {len(pending)} varlık yanıtsız, yalnızca bunlar tekrar sorulacak.")

    missing = [s for s in symbols if s.upper() not in analysis_dict]
    print(f"   Gemini: {AI_STATS['calls']} istek, {AI_STATS['quota_errors']} kota hatası, "
          f"{AI_STATS['fallback_switches']} yedek model geçişi, {len(missing)} yanıtsız varlık.")

    for index, row in final_analysis_list.iterrows():
        final_analysis_list.at[index, "AI Action & Risk Warning"] = analysis_dict.get(str(row["Asset"]).upper(), AI_FALLBACK_LABEL)
        final_analysis_list.at[index, "AI_Conviction"] = conviction_dict.get(str(row["Asset"]).upper(), np.nan)

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
            "regime": row.get("Regime") or pd.NA,
            "macro_impact": _round_or_na(row.get("Macro_Impact"), 2),
            "macro_driver": row.get("Macro_Driver") or pd.NA,
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
    # Model yanıtı yoksa "Tut" sayılmasın; eski kayıtlardaki yedek etiket de dahil
    if upper.startswith("NO SIGNAL") or "AI ISSUE" in upper or upper.startswith("DATA ERROR"):
        return NO_SIGNAL, ""
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
    macro_flag, driver = row.get("Macro_Flag"), str(row.get("Macro_Driver") or "").lower()
    if macro_flag == "Baskı":
        badges.append(f"🌪 {driver} {_num(row.get('Macro_Impact'), 1, sign=True)}%")
    elif macro_flag == "Destek":
        badges.append(f"🌤 {driver} {_num(row.get('Macro_Impact'), 1, sign=True)}%")
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

SCORE_ARROWS = {-2: "▼▼", -1: "▼", 0: "·", 1: "▲", 2: "▲▲"}

def _regime_line(macro):
    reg = macro["regime"]
    vix = f" · VIX {_num(reg['vix'])}" if pd.notna(reg.get("vix")) else ""
    spx = ""
    if reg.get("spx_trend") is not None:
        spx = " · S&P trend " + ("üstünde" if reg["spx_trend"] else "altında")
    return f"{reg['icon']} <b>Rejim: {reg['label']}</b>{vix}{spx}"

def _factor_panel_block(macro):
    """Faktör, bu haftaki gerçekleşen hareket ve haberlerden çıkan beklenen yön."""
    panel = macro.get("panel")
    if panel is None or panel.empty:
        return ""
    rows = []
    for _, p in panel.iterrows():
        move = f"{_num(p['week_move'], 0, sign=True)} bp" if p["kind"] == "bp" else f"{_num(p['week_move'], 1, sign=True)}%"
        rows.append((p["factor"], move, SCORE_ARROWS.get(macro["scores"].get(p["factor"], 0), "·")))
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    table = f"{'Faktör'.ljust(w1)}  {'Hafta'.rjust(w2)}  Haber\n" + "\n".join(
        f"{a.ljust(w1)}  {b.rjust(w2)}  {c}" for a, b, c in rows)
    movers = [f"{k}: {v}" for k, v in macro.get("reasons", {}).items() if macro["scores"].get(k) and v]
    extra = ("\n" + "\n".join(f"<i>• {_esc(m)}</i>" for m in movers[:4])) if movers else ""
    return f"📡 <b>Küresel faktörler</b>\n<pre>{_esc(table)}</pre>{extra}"

def _macro_blocks(macro):
    blocks = [_regime_line(macro)]
    panel = _factor_panel_block(macro)
    if panel:
        blocks.append(panel)
    blocks.append(f"🌍 <b>Piyasa notu</b>\n{_esc(macro['note'])}")
    return blocks

HEALTH_STYLE = {  # durum -> (simge, başlık), sıra önemlidir
    "exit": ("🔴", "Çıkışı değerlendir"),
    "warn": ("⚠️", "Erken uyarı"),
    "weak": ("🟠", "Zayıf, izlemede"),
    "ok": ("✅", "Sağlam"),
}

def _portfolio_block(health, compact=False):
    """Portföy, durum gruplarına ayrılmış halde. compact=True yalnızca sorunlu varlıkları gösterir."""
    if not health:
        return ""
    lines = [f"💼 <b>Portföyüm</b> · {len(health)} varlık"]
    for status, (icon, title) in HEALTH_STYLE.items():
        group = [h for h in health if h["status"] == status]
        if not group:
            continue
        if compact and status == "ok":
            lines.append(f"\n{icon} <b>{title}</b> ({len(group)}): {_esc(', '.join(h['symbol'] for h in group))}")
            continue
        lines.append(f"\n{icon} <b>{title}</b> ({len(group)})")
        for h in group:
            sig, reason = parse_signal(h.get("signal"))
            sig_icon = f"  {sig[1]}" if sig[0] else ""
            mom = f"  {_num(h['momentum'], 1, sign=True)}%" if h.get("momentum") is not None and pd.notna(h.get("momentum")) else ""
            lines.append(f"<b>{_esc(h['symbol'])}</b> · {_esc(h['name'])}  {_num(h['price'], 2)}{mom}{sig_icon}")
            detail = " · ".join(h["warnings"]) if h["warnings"] else reason
            if detail:
                lines.append(f"      <i>{_esc(detail)}</i>")
    return "\n".join(lines)

def _trade_plan_block(plan):
    cap = plan["capacity"]
    lines = ["🎯 <b>İşlem planı</b>"]
    for e in plan["exits"]:
        lines.append(f"🔴 <b>Sat: {_esc(e['symbol'])}</b>  {_num(e['ret'], 1, sign=True)}%  <i>{_esc(', '.join(e['why']))}</i>")
    for h in plan["holds"]:
        lines.append(f"🟡 Tut: {_esc(h['symbol'])}  {_num(h['ret'], 1, sign=True)}%")
    if plan["picks"]:
        for c in plan["picks"]:
            sig, reason = parse_signal(c["signal"])
            lines.append(f"🟢 <b>Al: {_esc(c['symbol'])}</b>  {_num(c['price'], 2)} · ~{_num(cap['size'], 0)} CHF · güven {c['conviction']}/5")
            if reason:
                lines.append(f"      <i>{_esc(reason)}</i>")
    else:
        why = "rejim risk-off" if plan["regime"] == "Risk-off" else (
            "açık öneri sınırı dolu" if plan["slots"] == 0 else "iki onayı birden alan varlık yok")
        lines.append(f"⏸️ Yeni alım yok — {why}.")
    lines.append(f"<i>Kural filtresini geçen {plan['gate_passed']}, model de onaylayan {plan['model_confirmed']} varlık. "
                 f"Rebalance başına en fazla {cap['n']} alım · gidiş-dönüş maliyeti ≈ %{_num(cap['cost_pct'], 1)} · "
                 f"en az {cap['min_hold_weeks']} hafta tutma. Core'dan bir satış ≈ %{_num(pl.FEE_PCT_PER_SIDE + pl.SPREAD_PCT, 2)}.</i>")
    rp = pl.recovery_path()
    lines.append(f"💰 {_num(rp['current'], 0)} → {_num(rp['target'], 0)} CHF · gereken {_num(rp['need_pct'], 1, sign=True)}% · "
                 f"%1/hafta ile ≈ {rp['weeks']} hafta\n"
                 f"<i>Yatırımda {_num(pl.INVESTED_CHF, 0)} · nakit {_num(pl.CASH_CHF, 0)} CHF</i>")
    return "\n".join(lines)

def build_daily_message(macro, shock_alerts, health=None):
    blocks = [_header("Günlük izleme"), _regime_line(macro)]
    pb = _portfolio_block(health, compact=True)
    if pb:
        blocks.append(pb)
    blocks += _macro_blocks(macro)[1:] + [_shock_block(shock_alerts)]
    return "\n\n".join(blocks)

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
    df = df[df["sig"].notna()]  # model yanıtı olmayan kayıtlar karneye girmez
    if df.empty:
        return None
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

def build_full_report_messages(macro, final_report_df, history_df, shock_alerts, health=None, plan=None):
    df = final_report_df.copy()
    df["_sig"] = df["AI Action & Risk Warning"].map(lambda t: parse_signal(t)[0][0])
    core = df[df["Category"] == "Core Foundation"]
    sats = df[df["Category"] != "Core Foundation"]

    # 1) Portföy ve işlem planı en üstte
    first = [_header("Rebalance raporu"), _regime_line(macro)]
    if health:
        first.append(_portfolio_block(health))
    if plan:
        first.append(_trade_plan_block(plan))
    messages = _chunk_lines("\n\n".join(first).split("\n"), "")
    messages[0] = messages[0].lstrip("\n")

    # 2) Piyasa: sinyal dağılımı, küresel faktörler, şoklar
    messages.append("\n\n".join(
        [f"🧭 <b>Bugünün sinyalleri</b>\n{_signal_counts_line(df)}\n<i>Fiyat · ▲▼ trend · 4 haftalık momentum · 🌪/🌤 makro etki</i>"]
        + _macro_blocks(macro)[1:] + [_shock_block(shock_alerts)]))

    # Portföy bloğu yoksa (eski çağrı) çekirdek listesi ayrı mesaj olarak kalır
    if not health:
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

def daily_core_health(macro):
    """Rebalance dışı günlerde portföyün erken uyarı taraması (fiyat + makro duyarlılık)."""
    impact = {}
    try:
        batch = batch_download_data(CORE_ASSETS)
        betas = ml.estimate_betas(batch, CORE_ASSETS, macro["factor_changes"])
        imp = ml.macro_impact(betas, macro["scores"])
        impact = imp["impact"].to_dict() if not imp.empty else {}
    except Exception:
        pass
    return pl.assess_core(CORE_ASSETS, impact)

if __name__ == "__main__":
    try:
        print("\n🚀 ALPHAGUARD SİSTEMİ BAŞLATILIYOR...")
        watchlist = read_portfolio("portfolio.csv")
        watchlist = [s for s in watchlist if s not in CORE_ASSETS] if watchlist else []
        
        all_monitored_symbols = list(set(CORE_ASSETS + watchlist))

        # 1. MAKRO & DEFCON ŞOK İZLEMESİ
        news_reader = ml.NewsReader()
        macro = ml.build_macro_context(news_reader, secure_ai_query)
        macro_note = macro["note"]
        print(f"Rejim: {macro['regime']['label']} | Makro skorlar: {macro['scores']}")
        shock_alerts = defcon_shock_monitor(all_monitored_symbols, macro_note, news_reader)

        # 2. REBALANCE
        if is_rebalance_day():
            final_report = dual_momentum_and_risk_analysis(watchlist, macro, news_reader)
            pd.set_option('display.max_colwidth', None)
            
            history_df = update_realized_returns(load_signal_history())
            history_df = append_new_signals(history_df, final_report)
            cols = [c for c in history_df.columns if c in HISTORY_COLUMNS or "1m" in c or "3m" in c]
            history_df[cols].to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
            
            print(generate_accuracy_summary(history_df))

            core_rows = {}
            for _, r in final_report[final_report["Category"] == "Core Foundation"].iterrows():
                d = r.to_dict()
                d["momentum"] = r.get(MOMENTUM_COL)
                core_rows[r["Asset"]] = d
            impact = final_report.set_index("Asset")["Macro_Impact"].to_dict() if "Macro_Impact" in final_report else {}
            health = pl.assess_core(CORE_ASSETS, impact, core_rows)
            sats = final_report[final_report["Category"] != "Core Foundation"]
            plan = pl.build_trade_plan(sats, macro["regime"]["label"], MOMENTUM_COL, RETURN_1W_COL)
            pl.save_trade_plan(plan["plan_df"])

            report_messages = build_full_report_messages(macro, final_report, history_df, shock_alerts, health, plan)
            print("\n\n".join(report_messages))
            send_telegram_message(report_messages)
        else:
            health = daily_core_health(macro)
            report_text = build_daily_message(macro, shock_alerts, health)
            print(report_text)
            send_telegram_message(report_text)
            
        print("\n🏁 SİSTEM BAŞARIYLA TAMAMLANDI!")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR (Sistem Çöktü): {e}")
        raise e
