# AlphaGuard AI: konteyner çalıştırma örneği.
# Mevcut GitHub Actions akışından bağımsızdır, onu değiştirmez.
FROM python:3.12-slim

# Python çıktısını anında göster, .pyc dosyası üretme
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Önce sadece bağımlılıklar: kod değişince bu katman yeniden kurulmaz (build cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sonra kod
COPY trend_bot.py altdata_verifier.py portfolio.csv ./

# Gizli anahtarlar imaja GÖMÜLMEZ, çalıştırırken -e ile verilir.
# signals_history.csv için /data klasörü bir "volume" olarak bağlanabilir.
CMD ["python", "-u", "trend_bot.py"]
