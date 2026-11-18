# AlphaGuard AI'yi Docker ile çalıştırmak

Bu dosyalar GitHub Actions akışını değiştirmez. Aynı kodu izole bir konteynerde çalıştırmak içindir.

## Derleme
```
docker build -t alphaguard .
```

## Çalıştırma
Anahtarlar imaja girmez, çalışma anında verilir:
```
docker run --rm \
  -e GEMINI_API_KEY=... \
  -e TELEGRAM_TOKEN=... \
  -e TELEGRAM_CHAT_ID=... \
  alphaguard
```

## Neden
- Her ortamda aynı Python ve aynı paket sürümleri (requirements.txt sabit sürümlü).
- Bağımlılıklar ayrı bir katmanda, kod değişince yeniden kurulmaz.
- Gizli anahtarlar kodda ya da imajda değil, çalıştırırken verilir.
