import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import time

class InstitutionalAltDataValidator:
    def __init__(self, uptrend_symbols):
        """
        AlphaGuard'dan gelen UPTREND varlık listesini alır.
        """
        self.symbols = [str(s).strip().upper() for s in uptrend_symbols if str(s).strip()]
        
    def get_options_sentiment(self, ticker_obj, symbol):
        """
        Opsiyon Piyasası Put/Call Oranını (PCR) hesaplar.
        Yüksek Put/Call (>1.0) = Kurumsal hedge / Çöküş korkusu
        Düşük Put/Call (<0.7) = Güçlü boğa iştahı
        """
        try:
            exp_dates = ticker_obj.options
            if not exp_dates:
                return "Veri Yok (Opsiyon İşlemi Olmayabilir)"
            
            # En yakın vadeli opsiyon zincirini çek
            opt_chain = ticker_obj.option_chain(exp_dates[0])
            puts_vol = opt_chain.puts['volume'].sum()
            calls_vol = opt_chain.calls['volume'].sum()
            
            if pd.isna(puts_vol) or pd.isna(calls_vol) or calls_vol == 0:
                return "Hacim Yetersiz"
                
            pcr = puts_vol / calls_vol
            
            if pcr > 1.2:
                status = "⚠️ AŞIRI KORUMA / ÇÖKÜŞ ENDİŞESİ (Yüksek Put)"
            elif pcr < 0.6:
                status = "🚀 GÜÇLÜ BOĞA İŞTAHI (Yüksek Call)"
            else:
                status = "⚖️ DENGELİ / NÖTR"
                
            return f"Put/Call Oranı: {pcr:.2f} -> {status}"
        except Exception:
            return "Opsiyon Verisi Alınamadı"

    def get_insider_activity(self, ticker_obj):
        """
        Şirket içi kişilerin (CEO, CFO, Fonlar) son alım/satımlarını inceler.
        """
        try:
            insider_df = ticker_obj.insider_transactions
            if insider_df is None or insider_df.empty:
                return "İçeriden İşlem Verisi Yok"
            
            # Son 30 gündeki işlemleri filtrele
            # yfinance insider_df yapısında 'Start Date' veya benzeri tarih sütunu olur
            if 'Start Date' in insider_df.columns:
                insider_df['Start Date'] = pd.to_datetime(insider_df['Start Date'], errors='coerce')
                recent = insider_df[insider_df['Start Date'] >= (dt.datetime.now() - dt.timedelta(days=30))]
            else:
                recent = insider_df.head(5) # Tarih sütunu yoksa en son 5 işleme bak
                
            if recent.empty:
                return "Son 1 Ayda İçeriden İşlem Yok"
                
            # Metin içinde 'Sale' veya 'Purchase' geçen sütunları kontrol et
            text_data = recent.to_string().lower()
            sales_count = text_data.count('sale') + text_data.count('sell')
            buys_count = text_data.count('purchase') + text_data.count('buy')
            
            if sales_count > buys_count:
                return f"🔴 İçeriden Satış Baskısı Ağırlıklı (Satış: {sales_count}, Alış: {buys_count})"
            elif buys_count > sales_count:
                return f"🟢 İçeriden Alım Desteği Var (Alış: {buys_count}, Satış: {sales_count})"
            else:
                return "Nötr / Karışık İçeriden İşlemler"
        except Exception:
            return "Insider Verisi Okunamadı"

    def get_analyst_consensus(self, ticker_obj):
        """
        Son ayın analist tavsiye sayılarını (strongBuy, buy, hold, sell, strongSell) okur.
        Kolon adlarına değil değerlere bakar; analist iyimserliği nedeniyle SAT eşiği daha düşüktür.
        """
        try:
            rec = ticker_obj.recommendations
            if rec is None or rec.empty:
                return "Analist Verisi Yok"
            needed = ["strongBuy", "buy", "hold", "sell", "strongSell"]
            if not all(c in rec.columns for c in needed):
                return "Analist Verisi Okunamadı"
            if "period" in rec.columns and (rec["period"] == "0m").any():
                row = rec[rec["period"] == "0m"].iloc[0]
            else:
                row = rec.iloc[0]
            counts = row[needed].astype(float).fillna(0)
            total = counts.sum()
            if total < 3:
                return "Yetersiz Analist Kapsamı"
            buy_share = (counts["strongBuy"] + counts["buy"]) / total
            sell_share = (counts["sell"] + counts["strongSell"]) / total
            detail = f"(Al %{buy_share * 100:.0f}, Sat %{sell_share * 100:.0f}, n={int(total)})"
            if sell_share >= 0.25:
                return f"🔴 Kurumsal Tavsiye: SAT Baskısı {detail}"
            if buy_share >= 0.60:
                return f"🟢 Kurumsal Tavsiye: AL {detail}"
            return f"🟡 Kurumsal Tavsiye: Tut / Nötr {detail}"
        except Exception:
            return "Analist Bilgisi Alınamadı"

    def audit_all_uptrend_assets(self):
        """
        AlphaGuard'dan gelen tüm UPTREND varlıkları kurum gözüyle tetkik eder.
        """
        print(f"\n========================================================")
        print(f"🕵️‍♂️ KURUMSAL AKILLI PARA (SMART MONEY) DOĞRULAMA MOTORU BAŞLADI")
        print(f"Taranacak Varlık Sayısı: {len(self.symbols)}")
        print(f"========================================================\n")
        
        audit_results = []
        
        for sym in self.symbols:
            print(f"Analiz Ediliyor: {sym}...")
            try:
                t_obj = yf.Ticker(sym)
                
                # 1. Opsiyon Duyarlılığı
                opt_sent = self.get_options_sentiment(t_obj, sym)
                
                # 2. İçeriden İşlemler (Insider)
                insider = self.get_insider_activity(t_obj)
                
                # 3. Analist Görüşü
                analyst = self.get_analyst_consensus(t_obj)
                
                audit_results.append({
                    "Asset": sym,
                    "Options Sentiment": opt_sent,
                    "Insider Activity": insider,
                    "Analyst Consensus": analyst
                })
                
                time.sleep(0.5) # Rate limit koruması
            except Exception as e:
                audit_results.append({
                    "Asset": sym,
                    "Options Sentiment": "Hata",
                    "Insider Activity": "Hata",
                    "Analyst Consensus": f"Hata: {e}"
                })
                
        return audit_results

if __name__ == "__main__":
    # Test amaçlı birkaç yükseliş trendindeki sembolle çalıştıralım
    test_uptrend_list = ["NVDA", "MSFT", "AAPL", "BTC-USD"]
    verifier = InstitutionalAltDataValidator(test_uptrend_list)
    results = verifier.audit_all_uptrend_assets()
    
    for r in results:
        print(f"\n📌 Varlık: {r['Asset']}")
        print(f"   - {r['Options Sentiment']}")
        print(f"   - {r['Insider Activity']}")
        print(f"   - {r['Analyst Consensus']}")
