# DividendAI — Smart SIP Advisor
**NSE India · 4 AI Models · 19 Financial Features · Live yfinance Data**

A pure-Python Streamlit dashboard that analyses NSE dividend stocks and recommends monthly SIP adjustments using an ensemble of four AI models.

---

## Quick Start

```bash
# 1. Install dependencies (~5 min first time for torch + transformers)
pip install -r requirements.txt

# 2. Launch dashboard
cd app.dividend-predictor
streamlit run app.py

# 3. Upload data/sample_sip.csv  →  click "Analyze Portfolio"
```

### Sample Input CSV (`data/sample_sip.csv`)
```csv
ticker,monthly_sip
HINDZINC.NS,3000
ITC.NS,2500
VEDL.NS,2000
TCS.NS,5000
INFY.NS,4000
```
Tickers without `.NS` are auto-appended (e.g. `HINDZINC` → `HINDZINC.NS`).

### Sample Output
```
HINDZINC  Score: 0.71  → Increase SIP  ₹3,000 → ₹3,400  (+₹400)
TCS       Score: 0.65  → Increase SIP  ₹5,000 → ₹5,700  (+₹700)
ITC       Score: 0.51  → Hold          ₹2,500 → ₹2,500  (±₹0)
INFY      Score: 0.47  → Hold          ₹4,000 → ₹4,000  (±₹0)
VEDL      Score: 0.38  → Reduce SIP    ₹2,000 → ₹1,600  (−₹400)
```

---

## How It Works — End-to-End Pipeline

```
Upload CSV (ticker, monthly_sip)
    ↓
Validate .NS suffix → batch_fetch (yfinance, safe try/except)
    ↓
build_feature_vector × n_tickers → impute_features (batch median fill)
    ↓
    ┌──────────────────┬──────────────────┬──────────────────┐
    ↓                  ↓                  ↓                  ↓
Random Forest      Pattern Memory      FinBERT          Base Rate
(70% weight)       (15% weight)       (10% weight)      (5% weight)
    ↓                  ↓                  ↓                  ↓
P(Reduce/Hold/     P(next quarter     Sentiment         0.55 constant
Increase) triple   dividend up)       score [0,1]       (NSE base rate)
    └──────────────────┴──────────────────┴──────────────────┘
                                ↓
              EnsembleCombiner.combine_batch()
              (weighted sum + weight redistribution for missing models)
                                ↓
              score ≥ 0.58 → Increase (+15%)
              score ≥ 0.42 → Hold     (±0%)
              score < 0.42 → Reduce   (−20%)
                                ↓
              PortfolioAdvisor → PortfolioRecommendation (rounded to ₹100)
                                ↓
              Streamlit: cards + dividend history + diagnostics
```

---

## File Structure

```
app.dividend-predictor/
├── app.py                        ← Streamlit UI + pipeline orchestration
├── config.py                     ← All tuneable constants (weights, thresholds, etc.)
├── requirements.txt
├── data/
│   └── sample_sip.csv
├── data_pipeline/
│   ├── fetcher.py                ← TickerData dataclass + safe yfinance wrappers
│   └── features.py               ← 19-feature vector + median imputation
├── models/
│   ├── random_forest.py          ← RF classifier (70%)
│   ├── lstm_model.py             ← MLP sequence model (15%)
│   ├── finbert_sentiment.py      ← FinBERT news scorer (10%)
│   └── ensemble.py               ← Weighted combiner + weight redistribution
├── analysis/
│   ├── backtest.py               ← Historical SIP sim + 1-year forward projection
│   └── portfolio.py              ← PortfolioAdvisor → PortfolioRecommendation
└── utils/
    └── logger.py                 ← Structured logging helpers
```

---

## Model 1 — Random Forest Classifier (70% weight)

### What It Does
Classifies each stock into **Increase / Hold / Reduce** based on 19 fundamental financial ratios. At 70%, this is the dominant signal.

### How It's Trained
- **Training data**: rolling annual snapshots from historical yfinance data.
  For each ticker with ≥5 years of dividends, one training row is created per year.
- **Label**: 3-year forward dividend CAGR:
  - `> 5%` → class 2 (Increase)
  - `< 0%` → class 0 (Reduce)
  - otherwise → class 1 (Hold)
- **Model**: `RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_leaf=2)`
- **Scaling**: `StandardScaler` applied before both training and prediction

### The 19 Input Features

| # | Feature | What It Measures |
|---|---|---|
| 1 | `roe` | Return on equity — profitability per shareholder ₹ |
| 2 | `debt_equity` | Financial leverage — <1 = safe payer |
| 3 | `dividend_yield` | Current yield from `yf.info` |
| 4 | `div_cagr_3y` | 3Y dividend CAGR — recent momentum |
| 5 | `div_cagr_5y` | 5Y dividend CAGR — longer trend |
| 6 | `payout_ratio` | % of earnings paid as dividends — sustainability |
| 7 | `eps_growth_3y` | Earnings per share growth — future capacity |
| 8 | `revenue_growth_3y` | Top-line strength |
| 9 | `ocf_margin` | Operating cashflow / revenue — cash generation |
| 10 | `fcf_yield` | Free cashflow / market cap — real ability to pay |
| 11 | `log_market_cap` | Log of market cap — size/stability proxy |
| 12 | `pe_ratio` | Price/Earnings — valuation |
| 13 | `pb_ratio` | Price/Book — asset backing |
| 14 | `current_ratio` | Current assets / liabilities — short-term liquidity |
| 15 | `interest_coverage` | EBIT / interest expense — debt servicing |
| 16 | `net_profit_margin` | Net income / revenue — bottom-line efficiency |
| 17 | `asset_turnover` | Revenue / total assets — operational efficiency |
| 18 | `sector_encoded` | Sector as integer (12 NSE sectors) |
| 19 | `consecutive_div_years` | Years of unbroken dividend payments — reliability |

### Sample Test Run

```python
import numpy as np
from models.random_forest import DividendRandomForest

rf = DividendRandomForest()

# Minimal mock training: 5 rows, 19 features (normally built from historical data)
X_train = np.array([
    # roe   d/e   yld  cag3 cag5 pay  eps  rev  ocf  fcf  lmc  pe   pb   cr   ic   npm  at   sec  div_yr
    [0.22, 0.4, 0.04, 0.12, 0.10, 0.35, 0.10, 0.08, 0.18, 0.05, 24.5, 14, 2.8, 2.1, 8.2, 0.15, 0.7, 2, 10],  # Increase
    [0.08, 1.8, 0.01, -0.05, 0.01, 0.80, -0.10, -0.02, 0.05, 0.01, 22.0, 30, 1.2, 0.9, 1.5, 0.03, 0.4, 1, 3],  # Reduce
    [0.15, 0.9, 0.03, 0.04, 0.05, 0.55, 0.05, 0.04, 0.12, 0.03, 23.0, 18, 2.0, 1.8, 4.5, 0.10, 0.6, 6, 7],  # Hold
    [0.25, 0.3, 0.05, 0.15, 0.12, 0.40, 0.14, 0.11, 0.20, 0.06, 25.0, 12, 3.5, 2.5, 10.0, 0.18, 0.8, 5, 12], # Increase
    [0.05, 2.2, 0.02, -0.08, -0.03, 0.90, -0.15, -0.05, 0.03, 0.01, 21.5, 35, 0.8, 0.7, 1.2, 0.02, 0.3, 1, 2], # Reduce
])
y_train = np.array([2, 0, 1, 2, 0])  # 2=Increase, 1=Hold, 0=Reduce

rf.train(X_train, y_train)

# Predict on a strong dividend stock (high ROE, low D/E, good FCF)
X_test = np.array([[0.20, 0.5, 0.04, 0.10, 0.08, 0.45, 0.09, 0.07, 0.16, 0.04, 24.0, 15, 2.5, 2.0, 7.0, 0.13, 0.65, 2, 9]])
probas = rf.predict_proba(X_test)
print(probas)
# Expected: [[P(Reduce), P(Hold), P(Increase)]]  e.g. [[0.05, 0.15, 0.80]]

importance = rf.get_feature_importance()
print(importance.head(5))
# Expected: top features like div_cagr_3y, roe, fcf_yield, payout_ratio, consecutive_div_years
```

**Output interpretation**: `probas[0][2]` = P(Increase) is passed to the ensemble. Values close to 1.0 = strong buy signal, close to 0.0 = reduce signal.

### Fallback behaviour
If fewer than 2 training rows can be assembled (very new listing, no dividend history), RF returns uniform `[0.33, 0.33, 0.33]` and its weight is redistributed to LSTM + FinBERT.

---

## Model 2 — Pattern Memory / MLP Sequence Model (15% weight)

### What It Does
Detects **dividend momentum patterns** from 10 years of quarterly dividend data. Predicts whether the next quarter's dividend will be higher than the current one.

### Architecture
`MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", early_stopping=True)`
— a scikit-learn Multi-Layer Perceptron (no TensorFlow dependency).

### How It Works
1. For each ticker, the annual dividend series is resampled to quarterly frequency.
2. A sliding window of `seq_len=8` quarters is extracted with 3 engineered features appended:
   - **Raw values** (normalised 0→1): 8 features
   - **Slope**: linear regression gradient over the window (rising vs falling trend)
   - **Momentum**: last quarter value − first quarter value
   - **Volatility**: std deviation of the window (irregular vs consistent payer)
3. Label: `1` if next quarter dividend > current quarter, else `0`.
4. The model trains on all sequences from **all tickers combined** (transfer learning across NSE stocks).

### Sample Test Run

```python
import numpy as np
import pandas as pd
from models.lstm_model import DividendLSTM

model = DividendLSTM(seq_len=8)

# Mock quarterly dividend series for 3 tickers (40 quarters = 10 years each)
np.random.seed(42)

# Consistently growing payer (HINDZINC-like)
hindzinc_divs = pd.Series(
    [0.5 + i * 0.08 + np.random.normal(0, 0.02) for i in range(40)],
    index=pd.date_range("2014-01-01", periods=40, freq="QS")
)

# Flat/declining payer (VEDL-like)
vedl_divs = pd.Series(
    [3.0 - i * 0.03 + np.random.normal(0, 0.1) for i in range(40)],
    index=pd.date_range("2014-01-01", periods=40, freq="QS")
)

# Irregular payer
itc_divs = pd.Series(
    [abs(np.random.normal(2.0, 0.8)) for _ in range(40)],
    index=pd.date_range("2014-01-01", periods=40, freq="QS")
)

div_dict = {"HINDZINC.NS": hindzinc_divs, "VEDL.NS": vedl_divs, "ITC.NS": itc_divs}

X, y = model.prepare_sequences(div_dict)
print(f"Sequences: X={X.shape}, positive rate={y.mean():.2f}")
# Expected: X=(~96, 11), positive rate ~0.55 for growing series

model.train(X, y)
print(f"Trained: {model.is_trained}, iterations: {model._model.n_iter_}")

# Predict on the growing series — should give high P(increase)
p_hindzinc = model.predict_proba(hindzinc_divs)
p_vedl     = model.predict_proba(vedl_divs)
print(f"HINDZINC P(increase): {p_hindzinc:.3f}")  # Expected: ~0.70+
print(f"VEDL     P(increase): {p_vedl:.3f}")      # Expected: ~0.35−
```

**Output interpretation**: Score > 0.5 = dividend momentum positive, < 0.5 = momentum weakening.

### Fallback behaviour
Requires `MIN_SEQUENCES=10` total sequences across all tickers. If unavailable (e.g. portfolio of 1 stock with only 2 years of dividends), the model returns `0.5` (neutral) and its 15% weight is redistributed proportionally to RF and FinBERT.

---

## Model 3 — FinBERT News Sentiment (10% weight)

### What It Does
Reads recent financial news headlines for each ticker via `yf.Ticker.news` and classifies them as **positive / neutral / negative** using the `ProsusAI/finbert` model (a BERT variant fine-tuned on financial text).

### How It Works
1. Fetch up to `NEWS_MAX_ARTICLES=10` headlines per ticker published within `NEWS_LOOKBACK_DAYS=5` days.
2. Pass each headline through the FinBERT pipeline (text-classification).
3. Each result is a `(label, confidence_score)` pair.
4. Aggregate contributions:
   - `positive` → `+confidence_score`
   - `negative` → `−confidence_score`
   - `neutral`  → `0`
5. Take the mean, then scale from `[−1, 1]` to `[0, 1]`: `score = (mean + 1) / 2`

### Sample Test Run

```python
from models.finbert_sentiment import FinBERTSentimentAnalyzer

analyzer = FinBERTSentimentAnalyzer()

# Example 1: All-positive news (dividend announcement, earnings beat)
positive_headlines = [
    "Hindustan Zinc declares record ₹21 per share dividend",
    "HINDZINC Q3 profit surges 32%, board recommends bumper payout",
    "Zinc prices rally on supply cuts, bullish for HINDZINC earnings",
]
score = analyzer.get_sentiment_score("HINDZINC.NS", positive_headlines)
print(f"All-positive score: {score:.3f}")   # Expected: ~0.85–0.95

# Example 2: All-negative news
negative_headlines = [
    "VEDL dividend cut as commodity prices crash, payout halved",
    "Vedanta faces debt crisis, analysts slash dividend forecasts",
    "Mining sector faces headwinds, VEDL under pressure",
]
score = analyzer.get_sentiment_score("VEDL.NS", negative_headlines)
print(f"All-negative score: {score:.3f}")   # Expected: ~0.05–0.20

# Example 3: Mixed news → neutral
mixed_headlines = [
    "ITC revenue grows, cigarette volume flat",
    "ITC dividend unchanged at ₹6.75 per share",
]
score = analyzer.get_sentiment_score("ITC.NS", mixed_headlines)
print(f"Mixed score: {score:.3f}")          # Expected: ~0.45–0.60

# Example 4: No news → neutral fallback
score = analyzer.get_sentiment_score("TICKER.NS", [])
print(f"No news score: {score:.3f}")        # Always exactly 0.5
```

### Offline Mode
The model is loaded with `TRANSFORMERS_OFFLINE=1` to prevent network calls to HuggingFace after the initial download. If the model is not yet cached (`~/.cache/huggingface/hub/`), it downloads ~400MB on first click.

### Fallback behaviour
- No headlines found → `0.5` (neutral, no signal)
- `transformers` not installed → `0.5` (neutral, weight redistributed)
- Model load failure → `0.5` (neutral, logged as warning)

---

## Model 4 — Base Rate Prior (5% weight)

### What It Does
A simple constant: `0.55`. This represents the long-run historical frequency with which NSE blue-chip dividend stocks raise their annual dividend. It acts as a **weak Bayesian prior** — a tiebreaker that nudges the ensemble slightly toward "Increase" when the data-driven models disagree.

```python
BASE_RATE = 0.55  # ~55% of NSE dividend stocks raise dividends year-on-year
```

No training. No prediction. Always available. Never changes.

---

## Ensemble Combiner — How All 4 Models Are Merged

### Weighted Sum
```python
ensemble_score = (0.70 × rf_P_increase) + (0.15 × lstm_score) + (0.10 × finbert_score) + (0.05 × 0.55)
```

### Decision Thresholds
```
ensemble_score ≥ 0.58  →  Increase SIP  (+15%)
ensemble_score ≥ 0.42  →  Hold          (±0%)
ensemble_score < 0.42  →  Reduce SIP    (−20%)
```

### Worked Example

```python
from models.ensemble import EnsembleCombiner, ModelOutputs
import numpy as np

combiner = EnsembleCombiner()

# Scenario A: Strong buy — all models agree
outputs_bullish = ModelOutputs(
    ticker="HINDZINC.NS",
    rf_proba=np.array([0.05, 0.15, 0.80]),   # RF: 80% P(Increase)
    lstm_proba=0.72,                           # Pattern: rising momentum
    finbert_score=0.88,                        # News: very positive
    rf_available=True, lstm_available=True, finbert_available=True,
)
result = combiner.combine(outputs_bullish)
print(f"Score: {result.ensemble_score:.3f}")        # ~0.79
print(f"Recommendation: {result.recommendation}")   # Increase
print(f"SIP adj: {result.sip_adjustment_pct:+.0%}") # +15%
print(f"Confidence: {result.confidence:.2f}")        # ~0.93 (models agree)

# Scenario B: Bearish — all models agree
outputs_bearish = ModelOutputs(
    ticker="VEDL.NS",
    rf_proba=np.array([0.70, 0.20, 0.10]),   # RF: 70% P(Reduce)
    lstm_proba=0.28,                           # Pattern: declining momentum
    finbert_score=0.15,                        # News: very negative
    rf_available=True, lstm_available=True, finbert_available=True,
)
result = combiner.combine(outputs_bearish)
print(f"Score: {result.ensemble_score:.3f}")        # ~0.17
print(f"Recommendation: {result.recommendation}")   # Reduce
print(f"SIP adj: {result.sip_adjustment_pct:+.0%}") # -20%

# Scenario C: LSTM unavailable (weight redistributed to RF + FinBERT)
outputs_no_lstm = ModelOutputs(
    ticker="NEWSTOCK.NS",
    rf_proba=np.array([0.10, 0.25, 0.65]),
    lstm_proba=0.5,    # value ignored since lstm_available=False
    finbert_score=0.70,
    rf_available=True, lstm_available=False, finbert_available=True,
)
result = combiner.combine(outputs_no_lstm)
# RF weight becomes 70/(70+10)*85% ≈ 74.4%  (LSTM's 15% redistributed)
# FinBERT weight becomes 10/(70+10)*85% ≈ 10.6%
print(f"Weights used: {result.model_weights_used}")
print(f"Score: {result.ensemble_score:.3f}")        # ~0.64 → Increase
```

### Confidence Score
```
confidence = 1 − stddev([rf_score, lstm_score, finbert_score])
```
- All 3 models agree (spread=0) → confidence ~1.0 (high conviction)
- Models split evenly → confidence ~0.7 (moderate)
- One outlier → confidence ~0.5 (low conviction)

---

## Portfolio Advisor — From Score to ₹ Recommendation

```python
from models.ensemble import EnsembleCombiner, ModelOutputs, EnsembleResult
from analysis.portfolio import PortfolioAdvisor
import pandas as pd
import numpy as np

# Build mock ensemble results
results = [
    EnsembleResult("HINDZINC.NS", ensemble_score=0.79, recommendation="Increase",
                   sip_adjustment_pct=0.15, confidence=0.93,
                   rf_score=0.80, lstm_score=0.72, finbert_score=0.88,
                   model_weights_used={"random_forest":0.70,"lstm":0.15,"finbert":0.10,"base":0.05}),
    EnsembleResult("VEDL.NS", ensemble_score=0.17, recommendation="Reduce",
                   sip_adjustment_pct=-0.20, confidence=0.88,
                   rf_score=0.10, lstm_score=0.28, finbert_score=0.15,
                   model_weights_used={"random_forest":0.70,"lstm":0.15,"finbert":0.10,"base":0.05}),
    EnsembleResult("ITC.NS", ensemble_score=0.52, recommendation="Hold",
                   sip_adjustment_pct=0.00, confidence=0.75,
                   rf_score=0.50, lstm_score=0.55, finbert_score=0.48,
                   model_weights_used={"random_forest":0.70,"lstm":0.15,"finbert":0.10,"base":0.05}),
]

sip_holdings = pd.DataFrame({
    "ticker": ["HINDZINC.NS", "VEDL.NS", "ITC.NS"],
    "monthly_sip": [3000, 2000, 2500]
})

advisor = PortfolioAdvisor(results)
recs = advisor.generate_recommendations(sip_holdings)

for r in recs:
    print(f"{r.ticker:<15} ₹{r.current_monthly_sip:,.0f} → ₹{r.recommended_monthly_sip:,.0f}  ({r.recommendation})")
    print(f"  Reasoning: {r.reasoning}")

# Output:
# HINDZINC.NS     ₹3,000 → ₹3,500  (Increase)   [3000 * 1.15 = 3450, rounded to ₹100]
#   Reasoning: High-conviction dividend growth signal (score 0.79, models agree).
# VEDL.NS         ₹2,000 → ₹1,600  (Reduce)
#   Reasoning: Dividend reduction risk detected (score 0.17); reduce exposure.
# ITC.NS          ₹2,500 → ₹2,500  (Hold)
#   Reasoning: Stable dividend expected (score 0.52); no clear catalyst.

totals = advisor.get_portfolio_totals(recs)
print(f"Total SIP: ₹{totals['current_total_sip']:,.0f} → ₹{totals['recommended_total_sip']:,.0f}")
# Total SIP: ₹7,500 → ₹7,600
```

**Rounding rule**: `round(current_sip × (1 + adj_pct), −2)` — always in ₹100 increments, minimum ₹100.

---

## Backtest & Dividend History

### Dividend History Chart (Section 04)
Shows the last 10 years of annual dividend per share for any selected stock:
- Light green bar = dividend increase vs prior year
- Light red bar = dividend cut
- Light indigo bar = first year (no prior to compare)
- Light gray bar = flat (unchanged)

### SIP Projection (available via `render_backtest_chart`)
The backtest engine (`analysis/backtest.py`) runs two simulations:
- **Historical section** (past 3 years): both "Baseline SIP" and "AI-Guided SIP" follow the **same path** using actual price data. This is for context — the AI recommendation is for the future, not the past.
- **Forward projection** (next 1 year, ~1 dividend cycle): Baseline uses the flat current SIP; AI-guided uses the recommended SIP (±15% / ±20%) applied from month 1. Future prices are simulated using the stock's own historical CAGR.

---

## Configuration (`config.py`)

| Parameter | Default | Effect |
|---|---|---|
| `MODEL_WEIGHTS` | `{rf:0.70, lstm:0.15, finbert:0.10, base:0.05}` | Adjustable via sidebar sliders |
| `THRESHOLDS.increase` | `0.58` | Score cutoff for Increase |
| `THRESHOLDS.hold` | `0.42` | Score cutoff for Hold (below = Reduce) |
| `SIP_ADJUSTMENTS.Increase` | `+0.15` | +15% SIP increase |
| `SIP_ADJUSTMENTS.Reduce` | `−0.20` | −20% SIP reduction |
| `HIST_YEARS` | `10` | Years of yfinance history to fetch |
| `BACKTEST_YEARS` | `3` | Years shown in historical chart section |
| `LSTM_SEQ_LEN` | `8` | Quarters per input window |
| `MIN_SEQUENCES` | `10` | Minimum sequences to train MLP |
| `NEWS_MAX_ARTICLES` | `10` | Max headlines per ticker |
| `NEWS_LOOKBACK_DAYS` | `5` | Headlines older than this are ignored |
| `RF_N_ESTIMATORS` | `200` | Number of decision trees |
| `RF_LABEL_CAGR_THRESHOLD` | `0.05` | 3Y div CAGR > 5% → Increase label |
| `SIP_ROUND_TO` | `100` | Recommended SIP rounded to nearest ₹100 |

---

## Missing Data Strategy

| Layer | Missing | Fallback |
|---|---|---|
| fetcher | any yf call fails | empty typed value + `fetch_errors` logged |
| features | field absent in `yf.info` | `NaN` → batch median imputed downstream |
| RF | < 2 training rows | uniform `[0.33, 0.33, 0.33]` + weight redistributed |
| Pattern MLP | < 10 sequences | `0.5` neutral + weight redistributed |
| FinBERT | no news / import error | `0.5` neutral + weight redistributed |
| ensemble | model unavailable | `_redistribute_weights()` rescales to 1.0 |
| backtest | price series < 12 months | ticker skipped with log warning |
| app | all-failed ticker | shown with "Data unavailable" banner |

---

## Architecture Decisions

### Why 4 models instead of 1?
A single ML model on ~30 NSE dividend stocks × 10 annual snapshots = 300 rows would severely overfit. Each specialist model contributes what it does best: fundamentals (RF), time-series patterns (MLP), real-time news (FinBERT), and a long-run prior (base rate).

### Why retrain every session?
No stale `.pkl` or `.h5` files. Every click trains on freshly downloaded yfinance data. `@st.cache_resource` prevents redundant retraining within the same session. Cost: ~30–90s per analysis. Benefit: always-current recommendations.

### Why MLP instead of LSTM?
The original design used TensorFlow LSTM, but TF crashes with AVX SIGABRT on Apple Silicon and adds a large dependency. The `sklearn.neural_network.MLPClassifier` achieves the same goal (learning temporal patterns from flattened sequences) with zero heavy dependencies and identical API ergonomics.

### Why FinBERT offline mode?
`TRANSFORMERS_OFFLINE=1` prevents the 400MB+ model from attempting network update checks on every run. After the first download (`~/.cache/huggingface/hub/`), all inference runs fully offline.

---

## Limitations

- **Retrain look-ahead in backtest**: historical section applies current AI recommendation retroactively — for context only, not predictive accuracy measurement.
- **Small training set**: RF trains on whatever annual snapshots exist for the uploaded tickers. 3–5 tickers may yield only 15–50 training rows.
- **NSE news sparsity**: `yf.Ticker.news` returns few or no results for most NSE stocks. FinBERT frequently falls back to neutral 0.5.
- **Not financial advice**: This is a research/learning tool. Always consult a SEBI-registered investment advisor before changing SIP allocations.

---

*Built with scikit-learn · HuggingFace Transformers (FinBERT) · yfinance · Streamlit · Plotly*
