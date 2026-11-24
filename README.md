# AlphaGuard AI

A rule-based core–satellite momentum monitor with LLM-assisted news screening, run on a schedule through GitHub Actions.

> Research prototype. It is not investment advice and it does not place orders.

## 1. Motivation

Trend-following rules are transparent and easy to audit, but they only see prices. A news shock, a cluster of insider sales or a sudden change in option positioning reaches a weekly price series late, if at all. This project tries a simple division of labour. A deterministic momentum filter decides which assets are eligible and how they are ranked. A language model then reads recent headlines and adjusts the label attached to each asset. The model never selects assets and never sets weights.

Because a short backtest on a hand-picked watchlist proves very little, the system logs its own signals every time it runs and scores them against realized returns later (Section 5). The log started on 3 August 2026.

The working objective is an average return of 1% per week. After the autumn 2026 sell-off the portfolio fell from about CHF 25,000 to about CHF 22,800 (CHF 14,500 invested, CHF 8,300 in cash). At 1% a week, recovering the 9.6% gap takes about ten weeks. It is used as a yardstick in the performance summary, not as a constraint: the system does not increase risk when it falls behind the target, because chasing a fixed reference point is exactly the kind of behaviour prospect theory predicts after losses (Kahneman & Tversky, 1979). Investors who are below a previous level tend to take larger risks to get back to it, the break-even effect described by Thaler & Johnson (1990). The trade plan is built to resist this: the number of trades is capped by cost, and the default is to do nothing.

## 2. Universe

The portfolio has two parts.

**Core.** The twelve positions actually held, defined in `portfolio_layer.py`: a gold ETF (`ZGLD.SW`), an S&P 500 ETF (`XSPX.SW`), NXP (`NXPI`), TSMC (`TSM`), a global aggregate bond ETF (`VAGX.SW`), a blockchain equity ETF (`BCHE.SW`), Sandisk (`SNDK`), a European defence ETF (`XDEF.DE`), Broadcom (`AVGO`), WTI crude oil futures (`CL=F`), Dogecoin (`DOGE-USD`) and Samsung Electronics (`SMSN.IL`). When Yahoo returns no data for a listing, the same product on another exchange is used. Core positions are not ranked. They are checked for signs that they should be sold (Section 3.8).

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

* **Equities:** option-implied measures, the direction of the latest eight insider transactions (see Lakonishok & Lee, 2001), and an analyst-consensus flag.

The option measures come from the nearest expiry that is 7 to 60 days away. Very short-dated contracts are skipped because their volume is dominated by short-horizon speculation. Contracts without open interest or with implausible implied volatility are dropped. Three values are computed:

1. the put/call volume ratio, kept only as context, since total public volume mixes opening and closing trades and is a weak proxy for the opening-buy volume studied by Pan & Poteshman (2006);
2. the implied-volatility spread, the open-interest-weighted difference between call and put implied volatility at the same strikes, for strikes within ±10% of spot (Cremers & Weinbaum, 2010);
3. the put skew, the implied volatility of the out-of-the-money put closest to K/S = 0.95 minus that of the at-the-money call (Xing, Zhang & Zhao, 2010).

Both papers work with cross-sectional sorts, so no absolute threshold is used. At each rebalance, assets in the lowest quintile of the spread or the highest quintile of the skew are flagged `OptRisk:High` (puts relatively expensive), the highest quintile of the spread is flagged `OptRisk:Low`, and the rest `OptRisk:Neutral`. No flag is set when fewer than ten assets have option data. The raw values and the flag are written to the signal log so that their predictive value can be tested separately from the model labels.
* **Crypto:** the Crypto Fear & Greed Index from alternative.me.
* **ETFs:** skipped, because equity-style insider and option data do not describe them well. A manual whitelist in `trend_bot.py` decides what counts as an ETF.

`altdata_verifier.py` runs the equity checks on their own for a list of symbols. It exists to test this module separately from the main pipeline.

### 3.6 Language-model layer

Assets are sent to Gemini 2.5 Flash (temperature 0.1, thinking disabled) in batches of 12. For each asset the prompt contains the trend label, the one-week return, the volume status, the alternative-data string, the expected macro impact (Section 3.7) and up to five headlines from the last seven days. Headlines are deduplicated. Many ETFs have no news feed on Yahoo, so for those the headlines of the underlying are used instead (gold futures for a gold ETF, the S&P 500 for an S&P tracker, and so on), or failing that the headlines of the fund's largest holdings. Such headlines are marked as coming from the underlying. The model must return one label from `STRONG BUY`, `ACCUMULATE`, `HOLD`, `TRIM` or `SELL`, followed by a justification of at most ten words. The prompt tells it to downgrade an asset by one level when insiders are selling or the option flag is `OptRisk:High`, and to upgrade it when insiders are buying. It also downgrades by one level when the expected macro impact is −1% or worse, and it may not use `STRONG BUY` in a risk-off regime.

The answer is requested as pipe-delimited plain text and split with ordinary string operations. An earlier JSON-based format failed often enough to leave gaps in the log, and the plain-text format is the simpler fix. Calls are spaced at least seven seconds apart to stay within the free-tier rate limit. On a quota error the bot waits for the delay the API suggests, and if the quota is still exhausted, or the daily quota is used up, it switches to a lighter Gemini model (`gemini-flash-lite-latest`, then `gemini-flash-latest`); a fallback that is not available is skipped. Thinking is disabled because the task needs no reasoning and thinking tokens count against the output limit, which used to cut answers short. If the API version does not accept that setting, it is dropped and the call is repeated. Assets missing from an answer are asked again in a smaller request, up to two more times, and only those still without an answer are marked `NO SIGNAL`. These rows are shown separately in the report and left out of the scorecard, so a missing answer is not counted as a `HOLD`. News-based sentiment scoring by language models is a growing research area (Tetlock, 2007, for the older dictionary-based approach; Lopez-Lira & Tang, 2023, for language models), but nothing here claims that the labels add predictive value. Testing that is what the log is for.

### 3.7 Macro layer

Global events such as a Fed decision, a sell-off in Nvidia, a move in Japanese rates or a war in the Middle East push most assets in the same direction. They are common factors. Common factors matter little for ranking assets against each other, but they matter for how much risk to take, and assets differ in how strongly they react to each one. The macro layer (`macro_layer.py`) measures these two effects separately. It follows the factor-model view of asset prices (Chen, Roll & Ross, 1986).

**Factor panel.** Eight weekly series are tracked: the S&P 500, the US 10-year yield, the VIX, the dollar index, USD/JPY, WTI crude, the Philadelphia semiconductor index and Bitcoin. USD/JPY stands in for Japanese rates and the yen carry trade, because Japanese government bond yields are not reliably available through Yahoo. Yield changes are measured in basis points, everything else in percent.

**Sensitivities.** For each asset, the sensitivity to each factor is estimated from the last 52 completed weeks as the expected weekly return, in percent, when the factor moves by one standard deviation (correlation times the asset's volatility). At least 26 overlapping weeks are required. The factors are correlated with each other, so they are estimated one at a time rather than in a joint regression. The resulting numbers describe exposure, not causation.

**Shock scores.** Up to thirty recent headlines on the factors and on Nvidia are sent to the model together with this week's factor moves. The model returns a score from −2 to +2 for the expected direction of each factor over the next week and a short note. It does not comment on individual assets. Factors the news says nothing clear about get a score of zero.

**Expected macro impact.** For each asset, the sensitivities are multiplied by the scores and summed. The result is the asset's expected weekly return effect from this week's global news, together with the factor that contributes most. An impact of −1% or worse is flagged as macro pressure, +1% or better as macro support. The threshold is the same size as the 1% weekly objective. For the inverse S&P ETF the sign is reversed.

**Regime.** The regime is `Risk-off` when the VIX is at 25 or above or the S&P 500 is below its 8-week average, `Gergin` (tense) when the VIX is between 20 and 25, and `Risk-on` otherwise. The VIX levels follow common practice (Whaley, 2000) and were not fitted to the data.

### 3.8 Portfolio monitoring and trade plan

**Early warnings for held positions.** The 8-week trend filter reacts late, so held positions are also checked on daily data, every day. Four price conditions are tested: the close is below its 20-day average; the 20-day average is lower than a week ago; the 20-day change is positive but the 10-day change is negative (momentum is fading); and the close is more than 2.5 average true ranges below the 22-day high, a volatility-based exit rule in the spirit of LeBeau's chandelier exit. Macro pressure, expensive puts, insider selling and a `TRIM` or `SELL` label from the model count as further warnings. A position whose weekly trend is still up gets an early warning when at least one price condition and one other warning are present. A position whose trend has already turned down is marked for exit when the model also labels it `TRIM` or `SELL`, or when two warnings are present. No rule can call a top before it happens; these checks shorten the delay and will produce false alarms.

**Number of trades.** The broker charges 0.65% of the traded amount on each side. With an assumed spread of 0.1%, a round trip costs about 1.5%, more than the 1% weekly objective. Because the fee is proportional, larger positions do not lower the cost rate; only trading less often does. The plan therefore opens at most one new position per rebalance, holds at most two at a time, splits the available cash into two equal slices, and keeps each position for at least three weeks, which spreads the cost to about 0.5% a week. Before the minimum holding period, a position is closed only when its trend has turned down and the model labels it `SELL`. Selling a held core position costs about 0.75%, which is shown next to the exit warnings. All of these values are parameters in `portfolio_layer.py`.

**Two independent confirmations.** A satellite is proposed only when it passes a rule-based screen and the model confirms it. The screen requires an uptrend, four-week momentum between 2% and 25% (very large recent gains tend to reverse at this horizon), a one-week gain below 8%, an expected macro impact above −0.5%, no expensive-put flag, no insider selling, and a regime other than risk-off. The model must label the asset `STRONG BUY` or `ACCUMULATE` with a conviction of at least 4 on a 1–5 scale. If nothing passes both, no trade is proposed. Not trading is the default.

**Exits.** Proposed positions are tracked in `trade_plan.csv`. At each rebalance an open proposal is closed when its trend turns down, when the model labels it `TRIM` or `SELL`, when it comes under macro pressure, or after six weeks without a buy label. Apart from a combined trend break and `SELL` label, these exits apply only after the three-week minimum holding period. Closed proposals keep their entry and exit prices, so the trade plan can be evaluated on its own.

## 4. Schedule and infrastructure

| Run | When | What happens |
|---|---|---|
| Daily | 09:40 Swiss time | Regime, early warnings for held positions, factor panel and macro note; shock monitor; short Telegram message |
| Rebalance | Tuesday and Thursday | Full ranking, model labels, signal log update, performance summary, report sent in parts |
| Extra run | 15:40 Swiss time, Tuesday and Thursday | Repeat at the US open |

Rebalancing moved from Monday/Friday to Tuesday/Thursday in November 2026. Tuesday avoids trading on weekend gaps and on Monday moves, which tend to reverse later in the week; Thursday still leaves a full day before the weekend and falls after Wednesday Fed decisions and before Friday payroll releases. The daily run stays on a daily schedule because news flow can change a trend within days. Report text is in Turkish.

The rebalance report arrives as four Telegram messages, ordered from summary to detail: (1) the regime, the held positions grouped into exit candidates, early warnings, weak and sound, and the trade plan with the path back to the recovery target; (2) the count of each label, a table of this week's factor moves next to the direction implied by the news, the macro note and any shock alerts; (3) the satellites grouped by label, with buy and sell groups listed first and `HOLD` reduced to a list of tickers; (4) a scorecard comparing the mean one-week return with the 1% objective and the hit rate with the share of positive returns across all signals. Each asset takes at most two lines: label, ticker, price, trend direction and four-week momentum, followed by the model's justification. Flags for expensive puts, insider selling and macro pressure or support are shown only when they apply.

The system needs no server. GitHub Actions runs the script and then commits `signals_history.csv` back to the repository, so the log is versioned and can be inspected in the commit history.

| File | Role |
|---|---|
| `trend_bot.py` | Main pipeline: data, signals, model calls, logging, Telegram |
| `macro_layer.py` | News reader, factor panel, sensitivities, macro shock scores, regime |
| `portfolio_layer.py` | Held positions, early warnings, trade capacity, trade plan |
| `trade_plan.csv` | Proposed trades with entry and exit |
| `altdata_verifier.py` | Stand-alone test of the equity alternative-data checks |
| `portfolio.csv` | Satellite watchlist |
| `signals_history.csv` | Append-only signal log with realized returns |
| `.github/workflows/trend_analizi.yml` | Scheduled run of the main pipeline |
| `.github/workflows/test_altdata.yml` | Manual run of the alternative-data test |

## 5. Evaluation protocol

At each rebalance one row per asset is appended to the log: date, price, trend label, four-week momentum and the model's label. On later runs, once 7 and 28 calendar days have passed, the current price is used to fill in the one-week and four-week realized returns. The evaluation happens at the first run after the horizon has elapsed, so the actual holding period can be somewhat longer than nominal.

A signal counts as a hit when the realized return is negative for `SELL` and `TRIM` labels and positive for every other label. The bot reports the number of evaluated signals, the mean return and the hit rate, and compares the mean one-week return with the 1% weekly objective.

The regime, the expected macro impact and its main factor are also logged. Once six to eight weeks of rebalances have been recorded, the macro layer will be tested on its own. The test asks whether the expected impact is related to realized one-week returns across assets, and whether model labels issued after the layer was added rank returns better than those issued before. If neither holds, the layer will be removed.

## 6. Limitations

* **Sample size and dependence.** The log covers a few weeks. Overlapping windows and strongly correlated assets mean the observations are far from independent. No significance tests are run.
* **Hit rate has no baseline.** In a rising market, a hit rate above 50% for non-sell labels is expected by chance. Returns are not benchmark-adjusted and ignore transaction costs.
* **Duplicate log entries.** Because the schedule includes two runs on each rebalance day, and manual runs are possible, the same asset can appear more than once on a given date. Any analysis of the log should first deduplicate on `(run_date, symbol)`.
* **Incomplete model coverage.** Rate limits still leave some assets without a label on busy days. Log entries before 20 November 2026 used a neutral fallback text for these cases; the scorecard recognises and excludes it.
* **Selection bias.** The watchlist was assembled by hand and is not a neutral universe.
* **Heuristic alternative data.** Insider activity is read by counting keywords in Yahoo's transaction table, and the analyst flag uses the current month's recommendation counts from Yahoo: `AL` when at least 60% of analysts rate the stock a buy, `SAT` when at least 25% rate it a sell (a lower bar, since sell ratings are rare; see Womack, 1996), and `TUT` otherwise. Coverage through `yfinance`, an unofficial interface, varies by exchange and can change without notice.
* **Model dependence.** Labels come from a non-deterministic model that sees at most five headlines per asset. Results may change with a different model version or prompt.
* **Macro layer.** Sensitivities come from one year of weekly data and are unstable around regime changes. The shock scores are the model's reading of headlines, not forecasts with a track record. Proxy headlines for ETFs describe the underlying market, not the fund.
* **Parameters.** The windows and thresholds have not been tuned or validated out of sample.

## 7. Running the code

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
export TELEGRAM_TOKEN=...
export TELEGRAM_CHAT_ID=...
python trend_bot.py
```

On GitHub the same three values are stored as repository secrets. Without a Gemini key the script still runs but returns placeholder labels. Without Telegram credentials it prints the report to the console. The report is only built on Tuesday and Thursday; on other days the script sends the daily monitoring message.

## References

Antonacci, G. (2014). *Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk.* McGraw-Hill.

Chen, N.-F., Roll, R., & Ross, S. A. (1986). Economic forces and the stock market. *Journal of Business, 59*(3), 383–403.

Cremers, M., & Weinbaum, D. (2010). Deviations from put-call parity and stock return predictability. *Journal of Financial and Quantitative Analysis, 45*(2), 335–367.

Faber, M. T. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management, 9*(4), 69–79.

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance, 48*(1), 65–91.

Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica, 47*(2), 263–291.

Lakonishok, J., & Lee, I. (2001). Are insider trades informative? *Review of Financial Studies, 14*(1), 79–111.

Lopez-Lira, A., & Tang, Y. (2023). Can ChatGPT forecast stock price movements? Return predictability and large language models. Working paper, SSRN.

Pan, J., & Poteshman, A. M. (2006). The information in option volume for future stock prices. *Review of Financial Studies, 19*(3), 871–908.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168.

Thaler, R. H., & Johnson, E. J. (1990). Gambling with the house money and trying to break even: The effects of prior outcomes on risky choice. *Management Science, 36*(6), 643–660.

Whaley, R. E. (2000). The investor fear gauge. *Journal of Portfolio Management, 26*(3), 12–17.

Womack, K. L. (1996). Do brokerage analysts' recommendations have investment value? *Journal of Finance, 51*(1), 137–167.

Xing, Y., Zhang, X., & Zhao, R. (2010). What does the individual option volatility smirk tell us about future equity returns? *Journal of Financial and Quantitative Analysis, 45*(3), 641–662.

---

*This repository is for academic and educational purposes. Nothing in it is financial advice.*
