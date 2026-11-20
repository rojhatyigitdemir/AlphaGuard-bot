"""
AlphaGuard makro katmanı.

Üç parçadan oluşur:
  1. Haber okuyucu: yfinance'in eski ve yeni haber formatını okur, tarihe göre süzer, tekrarları ayıklar.
     Kendi haberi olmayan ETF'ler için dayanak varlığın (veya en büyük pozisyonlarının) haberleri kullanılır.
  2. Makro faktör paneli: sekiz küresel faktörün haftalık hareketi ve her varlığın bu faktörlere duyarlılığı.
  3. Makro şok puanı: dil modeli küresel başlıkları okuyup her faktör için -2..+2 arası yön puanı verir.
     Varlığa beklenen makro etki = sum(duyarlılık x şok puanı).

Ortak şoklar (Fed kararı, Nvidia düşüşü, petrol) tüm varlıkları birlikte iter; bu yüzden göreli sıralamadan
çok risk seviyesini ve varlıklar arasındaki farklı duyarlılığı etkiler. Katman bu iki etkiyi ayrı ölçer.
"""

import datetime as dt
import re

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------
# 1. HABER OKUYUCU
# --------------------------------------------------------------------
NEWS_MAX_ITEMS = 5
NEWS_MAX_AGE_DAYS = 7

# Kendi haber akışı olmayan ETF'ler -> haberleri okunacak dayanak varlık (kök sembol, uzantısız).
# Listede olmayan ETF'ler için önce fonun en büyük pozisyonları denenir.
ETF_NEWS_PROXY = {
    "ZGLD": "GC=F",        # fiziki altın
    "ZSIL": "SI=F",        # fiziki gümüş
    "PLATC": "PL=F",       # platin
    "PALLC": "PA=F",       # paladyum
    "BNDW": "^TNX",        # küresel tahvil -> ABD 10Y faiz
    "VAGX": "^TNX",        # küresel tahvil (agregat)
    "VOO": "^GSPC",        # S&P 500
    "VUSD": "^GSPC",       # S&P 500
    "XSPS": "^GSPC",       # S&P 500 (ters ETF; yön modele ayrıca bildirilir)
    "VDEA": "^STOXX50E",   # gelişmiş Avrupa
    "MSE": "^STOXX50E",    # Euro Stoxx 50
    "CHDVD": "^SSMI",      # İsviçre temettü
    "IWMO": "URTH",        # MSCI World momentum
    "SBIO": "XBI",         # biyoteknoloji
    "PBE": "XBI",          # biyoteknoloji
}
INVERSE_ETFS = {"XSPS"}


def _article_title(item):
    """Eski format {'title': ...} ve yeni format {'content': {'title': ...}} ikisini de okur."""
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    return item.get("title") or content.get("title")


def _article_time(item):
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    raw = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            return pd.Timestamp(raw, unit="s", tz="UTC")
        ts = pd.Timestamp(raw)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return None


def _normalize(title):
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


class NewsReader:
    """Aynı çalıştırmada bir sembolün haberini bir kez çeker."""

    def __init__(self, max_items=NEWS_MAX_ITEMS, max_age_days=NEWS_MAX_AGE_DAYS):
        self.max_items = max_items
        self.max_age = pd.Timedelta(days=max_age_days)
        self._cache = {}

    def titles(self, symbol, max_items=None):
        max_items = max_items or self.max_items
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch(symbol)
        return self._cache[symbol][:max_items]

    def _fetch(self, symbol):
        try:
            news = yf.Ticker(symbol).news or []
        except Exception:
            return []
        now = pd.Timestamp.now(tz="UTC")
        seen, out = set(), []
        for item in news:
            title = _article_title(item)
            if not title:
                continue
            published = _article_time(item)
            if published is not None and now - published > self.max_age:
                continue
            key = _normalize(title)
            if key in seen:
                continue
            seen.add(key)
            out.append(title.strip())
        return out

    def asset_news(self, symbol, etf_list):
        """(başlıklar, kaynak) döndürür. ETF'lerde kendi haberi yoksa dayanak varlığa geçer."""
        own = self.titles(symbol)
        if own:
            return own, symbol
        base = symbol.upper().split(".")[0]
        if base not in etf_list:
            return [], None
        proxy = ETF_NEWS_PROXY.get(base)
        if proxy:
            return self.titles(proxy), proxy
        for holding in _top_holdings(symbol)[:3]:
            got = self.titles(holding, max_items=2)
            if got:
                return got, holding
        return [], None


def _top_holdings(symbol):
    try:
        holdings = yf.Ticker(symbol).funds_data.top_holdings
        if holdings is not None and not holdings.empty:
            return [str(s) for s in holdings.index.tolist()]
    except Exception:
        pass
    return []


# --------------------------------------------------------------------
# 2. MAKRO FAKTÖR PANELİ
# --------------------------------------------------------------------
# (ad, sembol, ölçü) — ölçü 'pct' yüzde değişim, 'bp' baz puan (getiri endeksleri için)
FACTORS = [
    ("Hisse", "^GSPC", "pct"),
    ("Faiz", "^TNX", "bp"),
    ("Volatilite", "^VIX", "pct"),
    ("Dolar", "DX-Y.NYB", "pct"),
    ("Yen", "JPY=X", "pct"),        # USD/JPY: Japonya faizi ve carry-trade çözülmesi için temsilci
    ("Petrol", "CL=F", "pct"),
    ("Çip/YZ", "^SOX", "pct"),
    ("Kripto", "BTC-USD", "pct"),
]
FACTOR_NAMES = [f[0] for f in FACTORS]
BETA_WINDOW = 52
BETA_MIN_OBS = 26
FACTOR_HEADLINE_SYMBOLS = ["^GSPC", "^TNX", "^VIX", "DX-Y.NYB", "JPY=X", "CL=F", "^SOX", "NVDA", "GC=F", "BTC-USD"]

REGIME_VIX_STRESS = 20.0
REGIME_VIX_RISK_OFF = 25.0
REGIME_SMA_WEEKS = 8
MACRO_FLAG_PCT = 1.0  # beklenen haftalık makro etki eşiği (haftalık %1 hedefle aynı ölçek)


def _closed_weekly_close(df_or_series):
    """Haftalık kapanış serisi; içinde bulunulan, bitmemiş hafta çıkarılır."""
    s = df_or_series.dropna()
    if s.empty:
        return s
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
    today = pd.Timestamp(dt.date.today())
    week_open = today - pd.Timedelta(days=today.weekday())
    if s.index[-1] >= week_open and len(s) > 1:
        s = s.iloc[:-1]
    s.index = s.index.to_period("W").start_time  # borsalar arası hizalama
    return s[~s.index.duplicated(keep="last")]


def _extract_close(batch, symbol):
    if batch is None:
        return pd.Series(dtype=float)
    try:
        if isinstance(batch.columns, pd.MultiIndex):
            if symbol not in batch.columns.get_level_values(0):
                return pd.Series(dtype=float)
            return batch[symbol]["Close"]
        return batch["Close"]
    except Exception:
        return pd.Series(dtype=float)


def _factor_change(close, kind):
    return close.diff() * 100 if kind == "bp" else close.pct_change() * 100


def build_factor_panel(period="2y"):
    """Faktörlerin haftalık değişim serileri ve son durum tablosu."""
    symbols = [f[1] for f in FACTORS]
    try:
        weekly = yf.download(symbols, period=period, interval="1wk", group_by="ticker", threads=True, progress=False)
    except Exception:
        weekly = None
    try:
        daily = yf.download(symbols, period="5d", interval="1d", group_by="ticker", threads=True, progress=False)
    except Exception:
        daily = None

    changes, rows = {}, []
    for name, sym, kind in FACTORS:
        closed = _closed_weekly_close(_extract_close(weekly, sym))
        if len(closed) < BETA_MIN_OBS:
            continue
        chg = _factor_change(closed, kind).dropna()
        changes[name] = chg
        last = _extract_close(daily, sym).dropna()
        current = float(last.iloc[-1]) if not last.empty else float(closed.iloc[-1])
        ref = float(closed.iloc[-1])
        move = (current - ref) * 100 if kind == "bp" else (current / ref - 1) * 100
        sd = chg.tail(BETA_WINDOW).std()
        rows.append({
            "factor": name, "symbol": sym, "kind": kind, "level": current,
            "week_move": move, "z": move / sd if sd and sd > 0 else np.nan,
            "sma_ok": current > closed.tail(REGIME_SMA_WEEKS).mean(),
        })
    return pd.DataFrame(changes), pd.DataFrame(rows)


def classify_regime(panel):
    """VIX seviyesi ve S&P'nin 8 haftalık ortalamaya göre konumu ile basit rejim etiketi."""
    if panel is None or panel.empty:
        return {"label": "Bilinmiyor", "icon": "⚪", "vix": np.nan, "spx_trend": None}
    by = panel.set_index("factor")
    vix = by.at["Volatilite", "level"] if "Volatilite" in by.index else np.nan
    spx_up = bool(by.at["Hisse", "sma_ok"]) if "Hisse" in by.index else None
    if (pd.notna(vix) and vix >= REGIME_VIX_RISK_OFF) or spx_up is False:
        label, icon = "Risk-off", "🔴"
    elif pd.notna(vix) and vix >= REGIME_VIX_STRESS:
        label, icon = "Gergin", "🟠"
    else:
        label, icon = "Risk-on", "🟢"
    return {"label": label, "icon": icon, "vix": vix, "spx_trend": spx_up}


def estimate_betas(batch, symbols, factor_changes):
    """
    Her varlık için tek faktörlü duyarlılık: faktör 1 standart sapma hareket ettiğinde
    varlığın beklenen haftalık getirisi (%). beta_sd = corr(r, f) * sd(r).
    Faktörler birbiriyle ilişkili olduğundan çoklu regresyon yerine tek tek tahmin edilir.
    """
    if factor_changes is None or factor_changes.empty:
        return pd.DataFrame()
    out = {}
    fac = factor_changes.tail(BETA_WINDOW * 2)
    for sym in symbols:
        r = _closed_weekly_close(_extract_close(batch, sym)).pct_change() * 100
        joined = pd.concat([r.rename("r"), fac], axis=1, join="inner").dropna(subset=["r"]).tail(BETA_WINDOW)
        if len(joined) < BETA_MIN_OBS:
            continue
        sd_r = joined["r"].std()
        out[sym] = {name: joined["r"].corr(joined[name]) * sd_r
                    for name in fac.columns if joined[name].notna().sum() >= BETA_MIN_OBS}
    return pd.DataFrame(out).T


# --------------------------------------------------------------------
# 3. MAKRO ŞOK PUANI
# --------------------------------------------------------------------
def score_macro_news(news_reader, ai_query, panel):
    """
    Küresel başlıkları tek çağrıda puanlatır. Model yalnızca faktör yönü verir (-2..+2),
    serbest yorum yerine yapılandırılmış çıktı ister; ayrıca kısa bir Türkçe not yazar.
    """
    headlines = []
    for sym in FACTOR_HEADLINE_SYMBOLS:
        for t in news_reader.titles(sym, max_items=4):
            if t not in headlines:
                headlines.append(t)
    headlines = headlines[:30]

    panel_txt = ""
    if panel is not None and not panel.empty:
        for _, p in panel.iterrows():
            unit = "bp" if p["kind"] == "bp" else "%"
            panel_txt += f"- {p['factor']}: bu hafta {p['week_move']:+.1f}{unit} (z={p['z']:+.1f})\n"

    scores = {name: 0 for name in FACTOR_NAMES}
    reasons = {}
    if not headlines:
        return scores, reasons, "Küresel piyasalarda belirgin bir haber akışı yok."

    prompt = f"""You are a chief economist. Weekly factor moves:
{panel_txt}
Headlines (last 7 days):
{chr(10).join('- ' + h for h in headlines)}

TASK 1. For each factor, score the expected direction of THE FACTOR ITSELF over the next week implied by the news:
-2 strong fall, -1 fall, 0 neutral, +1 rise, +2 strong rise.
Factors: {', '.join(FACTOR_NAMES)}
("Faiz" = US 10Y yield, "Volatilite" = VIX, "Yen" = USD/JPY rate, "Dolar" = DXY, "Çip/YZ" = semiconductor index.)
Use 0 when the news says nothing clear. Do not invent events.

TASK 2. Write a 60-word note in Turkish, plain prose, on what matters most this week.

FORMAT (exactly):
FAKTÖR | SCORE | max 8-word Turkish reason
NOT | note text
"""
    raw = ai_query(prompt) or ""
    note = ""
    for line in raw.split("\n"):
        if "|" not in line:
            continue
        parts = [p.strip().strip("*") for p in line.split("|")]
        key = parts[0]
        if key.upper().startswith("NOT"):
            note = " ".join(parts[1:]).strip()
            continue
        match = next((n for n in FACTOR_NAMES if n.lower() == key.lower()), None)
        if match and len(parts) >= 2:
            m = re.search(r"[-+−]?\d", parts[1].replace("−", "-"))
            if m:
                scores[match] = int(np.clip(int(m.group(0)), -2, 2))
                reasons[match] = parts[2] if len(parts) > 2 else ""
    return scores, reasons, note or "Makro not üretilemedi."


def macro_impact(betas, scores, inverse_symbols=()):
    """Varlık başına beklenen makro etki (%), en büyük katkıyı yapan faktör ve rüzgar bayrağı."""
    rows = {}
    if betas is None or betas.empty:
        return pd.DataFrame(columns=["impact", "driver", "flag"])
    s = pd.Series(scores, dtype=float)
    for sym, b in betas.iterrows():
        common = [c for c in b.index if c in s.index and pd.notna(b[c])]
        contrib = b[common] * s[common]
        impact = float(contrib.sum()) if common else np.nan
        if sym.upper().split(".")[0] in inverse_symbols:
            impact, contrib = -impact, -contrib
        driver = contrib.abs().idxmax() if common and contrib.abs().max() > 0 else ""
        flag = ""
        if pd.notna(impact) and impact <= -MACRO_FLAG_PCT:
            flag = "Baskı"
        elif pd.notna(impact) and impact >= MACRO_FLAG_PCT:
            flag = "Destek"
        rows[sym] = {"impact": impact, "driver": driver, "flag": flag}
    return pd.DataFrame(rows).T


def build_macro_context(news_reader, ai_query):
    """Günlük ve rebalance çalıştırmalarının ortak makro girdisi."""
    factor_changes, panel = build_factor_panel()
    regime = classify_regime(panel)
    scores, reasons, note = score_macro_news(news_reader, ai_query, panel)
    return {
        "factor_changes": factor_changes,
        "panel": panel,
        "regime": regime,
        "scores": scores,
        "reasons": reasons,
        "note": note.replace("\n", " "),
    }
