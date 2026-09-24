# AlphaGuard AI

A rule-based core–satellite momentum monitor with LLM-assisted news screening, run on a schedule through GitHub Actions.

> Research prototype. It is not investment advice and it does not place orders.

## 1. Motivation

Trend-following rules are transparent and easy to audit, but they only see prices. A news shock, a cluster of insider sales or a sudden change in option positioning reaches a weekly price series late, if at all. This project tries a simple division of labour. A deterministic momentum filter decides which assets are eligible and how they are ranked. A language model then reads recent headlines and adjusts the label attached to each asset. The model never selects assets and never sets weights.

Because a short backtest on a hand-picked watchlist proves very little, the system logs its own signals every time it runs and scores them against realized returns later (Section 5). The log started on 3 August 2026.

## 2. Universe

The portfolio has two parts.

**Core.** Eight fixed instruments that are followed regardless of their trend state: `O`, `BNDW`, `BTC-USD`, `ZGLD.SW`, `ZSIL.SW`, `SHEL`, `TSM`, `BCHE.SW`. They are meant to spread exposure across several asset classes, so the trend filter is reported for them but does not remove them.

**Satellite.** About seventy instruments read from the `Symbol` column of `portfolio.csv` (an export from a market-data terminal). Satellites are not held permanently. At each rebalance they are split by trend state and ranked.

The watchlist reflects my own interests and is not a random sample of any market. Section 6 comes back to this.

## 3. Method

### 3.1 Absolute momentum

Two years of weekly bars are downloaded from Yahoo Finance. An asset is labelled `UPTREND` when its latest price is above the 8-week simple moving average of completed weekly closes, and `DOWNTREND` otherwise. The bar for the current, unfinished week is left out of the average so that the reference level does not move during the week. This is the same idea as the moving-average timing rule in Faber (2007) and the absolute-momentum leg of Antonacci (2014), shortened to fit a weekly decision cycle.

### 3.2 Relative momentum

Assets in an uptrend are ranked by their four-week price change, highest first (Jegadeesh & Titman, 1993). Assets in a downtrend are kept in a separate list ordered by their most recent one-week return, which makes early reversals easy to spot. The 8-week and 4-week windows were chosen by judgement to match the trading horizon. They were not optimised.

### 3.3 Volume confirmation

The volume of the last completed week is compared with the mean of the four weeks before it. A change above +15% is read as increasing volume, below −15% as decreasing, and anything in between as stable. This flag is passed to the language model as context. It does not change the trend label.

### 3.4 Shock monitor

Once a day, every monitored symbol is checked for a price gap. A shock is flagged when today's high exceeds the previous close (or today's low falls below it) by more than 2.5 times the trailing 14-day mean daily range. This is an ATR-style measure, not Wilder's exact true range. The two most recent headlines per symbol are also sent to the language model, which is asked to separate genuinely disruptive news from noise.

### 3.5 Alternative data

Where the instrument type allows it, three market-based signals are attached to each asset:

* **Equities:** the put/call volume ratio for the nearest expiry (see Pan & Poteshman, 2006), the direction of the latest eight insider transactions (see Lakonishok & Lee, 2001), and a coarse analyst-recommendation flag.
* **Crypto:** the Crypto Fear & Greed Index from alternative.me.
* **ETFs:** skipped, because equity-style insider and option data do not describe them well. A manual whitelist in `trend_bot.py` decides what counts as an ETF.

`altdata_verifier.py` runs the equity checks on their own for a list of symbols. It exists to test this module separately from the main pipeline.

### 3.6 Language-model layer

Assets are sent to Gemini 2.5 Flash (temperature 0.1) in batches of 15. For each asset the prompt contains the trend label, the one-week return, the volume status, the alternative-data string and up to two recent headlines. The model must return one label from `STRONG BUY`, `ACCUMULATE`, `HOLD`, `TRIM` or `SELL`, followed by a justification of at most ten words. The prompt tells it to downgrade an asset when insiders are selling or the put/call ratio is above 1.2, and to upgrade it when insiders are buying.

The answer is requested as pipe-delimited plain text and split with ordinary string operations. An earlier JSON-based format failed often enough to leave gaps in the log, and the plain-text format is the simpler fix. If a call fails after three attempts, the affected assets receive a default "hold and monitor" label. News-based sentiment scoring by language models is a growing research area (Tetlock, 2007, for the older dictionary-based approach; Lopez-Lira & Tang, 2023, for language models), but nothing here claims that the labels add predictive value. Testing that is what the log is for.

## 4. Schedule and infrastructure

| Run | When | What happens |
|---|---|---|
| Daily | 09:40 Swiss time | Macro note from index, oil, rate and Bitcoin headlines; shock monitor; short Telegram message |
| Rebalance | Monday and Friday | Full ranking, model labels, signal log update, performance summary, report sent in parts |
| Extra run | 15:40 Swiss time, Monday and Friday | Repeat at the US open |

Fridays are included so that positions can be reviewed before the weekend close. The daily run stays on a daily schedule because news flow can change a trend within days. Report text is in Turkish.

The system needs no server. GitHub Actions runs the script and then commits `signals_history.csv` back to the repository, so the log is versioned and can be inspected in the commit history.

| File | Role |
|---|---|
| `trend_bot.py` | Main pipeline: data, signals, model calls, logging, Telegram |
| `altdata_verifier.py` | Stand-alone test of the equity alternative-data checks |
| `portfolio.csv` | Satellite watchlist |
| `signals_history.csv` | Append-only signal log with realized returns |
| `.github/workflows/trend_analizi.yml` | Scheduled run of the main pipeline |
| `.github/workflows/test_altdata.yml` | Manual run of the alternative-data test |

## 5. Evaluation protocol

At each rebalance one row per asset is appended to the log: date, price, trend label, four-week momentum and the model's label. On later runs, once 7 and 28 calendar days have passed, the current price is used to fill in the one-week and four-week realized returns. The evaluation happens at the first run after the horizon has elapsed, so the actual holding period can be somewhat longer than nominal.

A signal counts as a hit when the realized return is negative for `SELL` and `TRIM` labels and positive for every other label. The bot reports the number of evaluated signals, the mean return and the hit rate.

## 6. Limitations

* **Sample size and dependence.** The log covers a few weeks. Overlapping windows and strongly correlated assets mean the observations are far from independent. No significance tests are run.
* **Hit rate has no baseline.** In a rising market, a hit rate above 50% for non-sell labels is expected by chance. Returns are not benchmark-adjusted and ignore transaction costs.
* **Duplicate log entries.** Because the schedule includes two runs on Monday and Friday, and manual runs are possible, the same asset can appear more than once on a given date. Any analysis of the log should first deduplicate on `(run_date, symbol)`.
* **Incomplete model coverage.** When the model call fails, the fallback label is neutral, which mixes real signals with missing ones in the log.
* **Selection bias.** The watchlist was assembled by hand and is not a neutral universe.
* **Heuristic alternative data.** Insider activity is read by counting keywords in Yahoo's transaction table, and the analyst flag is coarse. Coverage through `yfinance`, an unofficial interface, varies by exchange and can change without notice.
* **Model dependence.** Labels come from a non-deterministic model that sees only two headlines per asset. Results may change with a different model version or prompt.
* **Parameters.** The windows and thresholds have not been tuned or validated out of sample.

## 7. Running the code

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
export TELEGRAM_TOKEN=...
export TELEGRAM_CHAT_ID=...
python trend_bot.py
```

On GitHub the same three values are stored as repository secrets. Without a Gemini key the script still runs but returns placeholder labels. Without Telegram credentials it prints the report to the console. The report is only built on Monday and Friday; on other days the script sends the daily monitoring message.

## References

Antonacci, G. (2014). *Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk.* McGraw-Hill.

Faber, M. T. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management, 9*(4), 69–79.

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance, 48*(1), 65–91.

Lakonishok, J., & Lee, I. (2001). Are insider trades informative? *Review of Financial Studies, 14*(1), 79–111.

Lopez-Lira, A., & Tang, Y. (2023). Can ChatGPT forecast stock price movements? Return predictability and large language models. Working paper, SSRN.

Pan, J., & Poteshman, A. M. (2006). The information in option volume for future stock prices. *Review of Financial Studies, 19*(3), 871–908.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168.

---

*This repository is for academic and educational purposes. Nothing in it is financial advice.*
