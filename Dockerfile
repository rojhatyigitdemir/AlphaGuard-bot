FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# paketler önce, cache için
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY trend_bot.py altdata_verifier.py portfolio.csv ./
CMD ["python", "-u", "trend_bot.py"]
