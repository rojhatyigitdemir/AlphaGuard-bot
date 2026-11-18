# Docker

Actions akışından bağımsız, aynı botu konteynerde çalıştırır.

## Build
    docker build -t alphaguard .

## Run
    docker run --rm -e GEMINI_API_KEY=... -e TELEGRAM_TOKEN=... -e TELEGRAM_CHAT_ID=... alphaguard

Anahtarlar imaja girmez, run sırasında verilir.
