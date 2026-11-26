"""
AlphaGuard portföy katmanı.

  1. Portföy tanımı: gerçekte tutulan varlıklar (core) ve Telegram'da görünen adları.
  2. Erken uyarı: haftalık trend kırılmadan önce zayıflamayı yakalamaya çalışan günlük göstergeler.
  3. İşlem planı: işlem ücreti nedeniyle az sayıda, iki bağımsız onaydan geçen alım önerisi
     ve önerilen pozisyonların çıkış takibi.

Hiçbir gösterge düşüşü önceden kesin olarak bilemez. Erken uyarı katmanının amacı, 8 haftalık
ortalamaya dayalı ana trend filtresinin gecikmesini kısaltmaktır; yanlış alarm üretmesi beklenir.
"""

import datetime as dt
import math
import os

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------
# 1. PORTFÖY
# --------------------------------------------------------------------
# Sıra Telegram'daki sırayı belirler. Yahoo'da işlem görmeyen ürünler için vekil sembol kullanılır.
PORTFOLIO = [
    ("ZGLD.SW", "Swisscanto Gold ETF AA"),
    ("XSPX.SW", "Xtrackers S&P 500 Swap"),
    ("NXPI", "NXP Semiconductors"),
    ("TSM", "TSMC"),
    ("VAGX.SW", "Vanguard Global Agg. Bond"),
    ("BCHE.SW", "Invesco Global Blockchain"),
    ("SNDK", "Sandisk"),
    ("XDEF.DE", "Xtrackers Europe Defence"),
    ("DOGE-USD", "Dogecoin"),
    ("CSSMI.SW", "iShares SMI (CH)"),
    ("TUR", "iShares MSCI Turkey ETF"),
]
PORTFOLIO_SYMBOLS = [p[0] for p in PORTFOLIO]
# Yahoo bir kotasyonu döndürmezse aynı ürünün başka borsadaki kotasyonu kullanılır
SYMBOL_FALLBACKS = {"XSPX.SW": "D5BM.DE", "XDEF.DE": "XDEF.MI", "VAGX.SW": "VAGX.L", "SMSN.IL": "005930.KS"}
PORTFOLIO_NAMES = dict(PORTFOLIO)

# --------------------------------------------------------------------
# 2. SERMAYE, MALİYET VE İŞLEM SAYISI
# --------------------------------------------------------------------
INVESTED_CHF = 14_500            # piyasadaki pozisyonlar (son bildirilen)
CASH_CHF = 8_300                 # nakit
PORTFOLIO_VALUE_CHF = INVESTED_CHF + CASH_CHF
RECOVERY_TARGET_CHF = 25_000     # geri dönülmek istenen seviye
FEE_PCT_PER_SIDE = 0.65          # komisyon: 1.000 CHF başına 6,5 CHF, alışta ve satışta ayrı ayrı
SPREAD_PCT = 0.10                # tek yönlü alış-satış makası tahmini (%)
MAX_OPEN_POSITIONS = 2           # aynı anda açık uydu önerisi
MAX_NEW_POSITIONS = 1            # rebalance başına yeni alım
MIN_HOLD_WEEKS = 3               # maliyet bu süreye yayılır; erken çıkış yalnızca sert sinyalde
MAX_HOLD_WEEKS = 6               # bu süreden sonra güç yoksa öneri kapanır


def round_trip_cost_pct():
    return 2 * FEE_PCT_PER_SIDE + 2 * SPREAD_PCT


def trade_capacity(cash=CASH_CHF):
    """
    Komisyon tutara orantılı olduğu için pozisyonu büyütmek maliyet oranını düşürmez; sınır işlem
    sıklığıdır. Gidiş-dönüş yaklaşık %1,5 tutar, yani haftalık %1 hedefin üstünde. Bu yüzden
    rebalance başına en fazla bir alım yapılır, pozisyon en az üç hafta tutulur ve nakit iki eşit
    dilime bölünür.
    """
    cost = round_trip_cost_pct()
    size = cash / MAX_OPEN_POSITIONS if MAX_OPEN_POSITIONS else 0.0
    return {"budget": cash, "n": MAX_NEW_POSITIONS, "size": size, "cost_pct": cost,
            "cost_per_week_pct": cost / MIN_HOLD_WEEKS, "min_hold_weeks": MIN_HOLD_WEEKS}


def recovery_path(current=PORTFOLIO_VALUE_CHF, target=RECOVERY_TARGET_CHF, weekly_pct=1.0):
    need = (target / current - 1) * 100
    weeks = math.ceil(math.log(target / current) / math.log(1 + weekly_pct / 100)) if current < target else 0
    return {"current": current, "target": target, "need_pct": need, "weeks": weeks}


# --------------------------------------------------------------------
# 3. ERKEN UYARI
# --------------------------------------------------------------------
SMA_FAST_DAYS = 20
ATR_DAYS = 14
CHANDELIER_ATR = 2.5   # 22 günlük zirveden 2,5 ATR geri çekilme (LeBeau'nun "chandelier" çıkışı)


STALE_DAYS = 4          # en yeni veriden bu kadar gün geride kalan seri bayat sayılır (hafta sonu + tatil payı)


def _last_day(df):
    ts = pd.Timestamp(df.index[-1])
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _history(sym):
    df = yf.Ticker(sym).history(period="6mo", interval="1d")
    return df.dropna(subset=["Close"]) if "Close" in df else df


def _daily_close_frame(symbols):
    try:
        data = yf.download(symbols, period="6mo", interval="1d", group_by="ticker", threads=True, progress=False)
    except Exception:
        data = None
    out = {}
    for sym in symbols:
        try:
            df = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(subset=["Close"])
            if len(df) < 40:
                df = _history(sym)
            if len(df) >= 40:
                out[sym] = df
        except Exception:
            try:
                df = _history(sym)
                if len(df) >= 40:
                    out[sym] = df
            except Exception:
                pass
    # Toplu indirmede bazı seriler (ör. hisse bölünmesinden sonra) eski bir tarihte kalabiliyor;
    # bu durumda son fiyat yanlış görünür. Bayat seriyi tekil sorguyla yenile, olmazsa kullanma.
    equity_days = [_last_day(f) for s, f in out.items() if not (s.endswith("-USD") or s.endswith("=F"))]
    if equity_days:
        newest = max(equity_days)
        for sym in list(out):
            if (newest - _last_day(out[sym])).days <= STALE_DAYS:
                continue
            try:
                df = _history(sym)
            except Exception:
                df = pd.DataFrame()
            if len(df) >= 40 and (newest - _last_day(df)).days <= STALE_DAYS:
                out[sym] = df
            else:
                print(f"   [Veri] {sym} günlük serisi {_last_day(out[sym]).date()} tarihinde kalmış; erken uyarıdan çıkarıldı.")
                del out[sym]
    return out


def price_warnings(df):
    """Günlük fiyattan erken zayıflama işaretleri. Liste ve son fiyat döndürür."""
    close, high, low = df["Close"].ffill(), df["High"].ffill(), df["Low"].ffill()
    warnings = []
    sma = close.rolling(SMA_FAST_DAYS).mean()
    if close.iloc[-1] < sma.iloc[-1]:
        warnings.append("20 günlük ortalamanın altında")
    if sma.iloc[-1] < sma.iloc[-6]:
        warnings.append("20 günlük ortalama aşağı dönüyor")
    roc10 = close.iloc[-1] / close.iloc[-11] - 1
    roc20 = close.iloc[-1] / close.iloc[-21] - 1
    if roc20 > 0 and roc10 < 0:
        warnings.append("momentum yavaşlıyor")
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_DAYS).mean().iloc[-1]
    peak = high.tail(22).max()
    if pd.notna(atr) and close.iloc[-1] < peak - CHANDELIER_ATR * atr:
        warnings.append("zirveden 2,5 ATR geri çekildi")
    return warnings, float(close.iloc[-1])


def assess_core(symbols, macro_impact=None, report_rows=None):
    """
    Her core varlık için durum: 'exit' (çıkışı değerlendir), 'warn' (erken uyarı), 'weak', 'ok'.
    report_rows (rebalance günü) model etiketi, opsiyon, insider ve haftalık trend bilgisini ekler.
    """
    frames = _daily_close_frame(symbols)
    for sym in symbols:
        alt = SYMBOL_FALLBACKS.get(sym)
        if sym not in frames and alt and alt != sym:
            frames.update({sym: f for s, f in _daily_close_frame([alt]).items()})
    results = []
    for sym in symbols:
        row = report_rows.get(sym, {}) if report_rows else {}
        warns, price = [], row.get("Price ($)", np.nan)
        price_based = 0
        if sym in frames:
            pw, price = price_warnings(frames[sym])
            warns += pw
            price_based = len(pw)
        mi = macro_impact.get(sym) if macro_impact is not None else None
        if mi is not None and pd.notna(mi) and mi <= -1.0:
            warns.append(f"makro baskı {mi:+.1f}%".replace(".", ","))
        if row.get("Opt_Flag") == "OptRisk:High":
            warns.append("put'lar pahalı")
        if "Ins:Satış" in str(row.get("AltData (Smart Money)", "")):
            warns.append("insider satışı")
        label = str(row.get("AI Action & Risk Warning", "")).upper()
        model_bearish = ("SELL" in label or "TRIM" in label) and "NO SIGNAL" not in label
        if model_bearish:
            warns.append("model: azalt/sat")
        trend = str(row.get("Absolute Trend", ""))
        trend_down = "DOWN" in trend
        if not trend and sym in frames:  # günlük çalıştırma: haftalık trend yerine 40 günlük ortalama
            c = frames[sym]["Close"].ffill()
            trend_down = c.iloc[-1] < c.tail(40).mean()

        if trend_down and (model_bearish or len(warns) >= 2):
            status = "exit"
        elif not trend_down and price_based >= 1 and len(warns) >= 2:
            status = "warn"
        elif trend_down:
            status = "weak"
        else:
            status = "ok"
        results.append({"symbol": sym, "name": PORTFOLIO_NAMES.get(sym, sym), "price": price,
                        "status": status, "warnings": warns, "trend_down": trend_down,
                        "signal": row.get("AI Action & Risk Warning", ""),
                        "momentum": row.get("momentum")})
    return results


# --------------------------------------------------------------------
# 4. İŞLEM PLANI
# --------------------------------------------------------------------
TRADE_PLAN_FILE = "trade_plan.csv"
TRADE_PLAN_COLUMNS = ["open_date", "symbol", "entry_price", "size_chf", "conviction", "reason",
                      "status", "close_date", "exit_price", "return_pct"]
MOM_MIN, MOM_MAX = 2.0, 25.0   # 4 haftalık momentum aralığı; aşırı yükselenler kısa vadede geri döner
RET_1W_MAX = 8.0               # son bir haftada %8'den fazla yükselene binilmez
MIN_CONVICTION = 4             # modelin 1-5 arası güveni


def quant_gate(row, regime):
    """Birinci anahtar: kurala dayalı ön eleme. (geçti mi, gerekçe)"""
    reasons = []
    if "UP" not in str(row.get("Absolute Trend", "")):
        reasons.append("trend aşağı")
    mom = row.get("mom")
    if pd.isna(mom) or not (MOM_MIN <= mom <= MOM_MAX):
        reasons.append("momentum aralık dışında")
    r1 = row.get("ret1w")
    if pd.notna(r1) and r1 > RET_1W_MAX:
        reasons.append("son hafta aşırı yükseldi")
    mi = row.get("Macro_Impact")
    if pd.notna(mi) and mi <= -0.5:
        reasons.append("makro baskı")
    if row.get("Opt_Flag") == "OptRisk:High":
        reasons.append("put'lar pahalı")
    if "Ins:Satış" in str(row.get("AltData (Smart Money)", "")):
        reasons.append("insider satışı")
    if regime == "Risk-off":
        reasons.append("rejim risk-off")
    return (not reasons), reasons


def load_trade_plan():
    if os.path.exists(TRADE_PLAN_FILE):
        try:
            df = pd.read_csv(TRADE_PLAN_FILE)
            for c in TRADE_PLAN_COLUMNS:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[TRADE_PLAN_COLUMNS].astype(object)
        except Exception:
            pass
    return pd.DataFrame(columns=TRADE_PLAN_COLUMNS, dtype=object)


def build_trade_plan(sat_df, regime, mom_col, ret_col, plan_df=None):
    """
    Açık önerileri günceller (çıkış kontrolü) ve yeni alımları seçer.
    İkinci anahtar model etiketidir: yalnızca STRONG BUY / ACCUMULATE ve güven >= 4.
    """
    today = pd.Timestamp(dt.date.today())
    plan = (load_trade_plan() if plan_df is None else plan_df.copy()).astype(object)
    cap = trade_capacity()
    by_sym = sat_df.set_index("Asset")

    exits, holds = [], []
    for idx, p in plan[plan["status"] == "open"].iterrows():
        sym = p["symbol"]
        if sym not in by_sym.index:
            holds.append({"symbol": sym, "note": "veri yok", "ret": np.nan})
            continue
        r = by_sym.loc[sym]
        price = float(r["Price ($)"])
        ret = (price / float(p["entry_price"]) - 1) * 100 if float(p["entry_price"] or 0) else np.nan
        label = str(r.get("AI Action & Risk Warning", "")).upper()
        weeks = (today - pd.Timestamp(p["open_date"])).days / 7
        why = []
        trend_broken = "DOWN" in str(r.get("Absolute Trend", ""))
        model_sell = "SELL" in label and "NO SIGNAL" not in label
        if trend_broken and model_sell:  # sert sinyal: tutma süresini beklemez
            why.append("trend kırıldı ve model sat diyor")
        elif weeks >= MIN_HOLD_WEEKS:
            if trend_broken:
                why.append("trend kırıldı")
            if ("SELL" in label or "TRIM" in label) and "NO SIGNAL" not in label:
                why.append("model azalt/sat")
            if r.get("Macro_Flag") == "Baskı":
                why.append("makro baskı")
            if weeks >= MAX_HOLD_WEEKS and not ("STRONG BUY" in label or "ACCUMULATE" in label):
                why.append(f"{MAX_HOLD_WEEKS} hafta doldu, güç kalmadı")
        if why:
            plan.loc[idx, ["status", "close_date", "exit_price", "return_pct"]] = ["closed", today.date(), price, round(ret, 2)]
            exits.append({"symbol": sym, "ret": ret, "why": why})
        else:
            holds.append({"symbol": sym, "ret": ret, "note": f"{weeks:.0f}. hafta"})

    open_after = int((plan["status"] == "open").sum())
    slots = max(0, min(cap["n"], MAX_OPEN_POSITIONS - open_after))
    candidates, rejected_by_model = [], 0
    held = set(plan.loc[plan["status"] == "open", "symbol"])
    for sym, r in by_sym.iterrows():
        if sym in held:
            continue
        row = dict(r)
        row["mom"], row["ret1w"] = r.get(mom_col), r.get(ret_col)
        ok, _ = quant_gate(row, regime)
        if not ok:
            continue
        label = str(r.get("AI Action & Risk Warning", "")).upper()
        conv = r.get("AI_Conviction")
        if not (("STRONG BUY" in label or "ACCUMULATE" in label) and pd.notna(conv) and conv >= MIN_CONVICTION):
            rejected_by_model += 1
            continue
        candidates.append({"symbol": sym, "price": float(r["Price ($)"]), "conviction": int(conv),
                           "mom": float(row["mom"]), "signal": r.get("AI Action & Risk Warning", ""),
                           "strong": "STRONG BUY" in label})
    candidates.sort(key=lambda c: (c["conviction"], c["strong"], -abs(c["mom"] - 10)), reverse=True)
    picks = candidates[:slots]
    for c in picks:
        plan.loc[len(plan)] = [today.date(), c["symbol"], c["price"], round(cap["size"]), c["conviction"],
                               str(c["signal"])[:120], "open", pd.NA, pd.NA, pd.NA]
    return {
        "picks": picks, "exits": exits, "holds": holds, "capacity": cap, "slots": slots,
        "gate_passed": len(candidates) + rejected_by_model, "model_confirmed": len(candidates),
        "regime": regime, "plan_df": plan,
    }


def save_trade_plan(plan_df):
    plan_df.to_csv(TRADE_PLAN_FILE, index=False, encoding="utf-8-sig")
