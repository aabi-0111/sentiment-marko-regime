# Sentiment-Driven Markov Regime Switching for Dynamic Portfolio Risk Management

This is a Python-based finance research project. It uses market returns and investor sentiment to study market regimes. The goal is to manage portfolio exposure. The project combines Bitcoin price data (`BTC-USD`) with the Alternative.me Crypto Fear & Greed Index. It builds a regime-switching model to evaluate a strategy against a buy-and-hold benchmark.

## Overview

Financial markets do not stay in one state forever. They shift between phases and volatile ones. A regime-switching model helps detect these shifts. This allows for risk management. This project looks at whether combining investor sentiment with a model can improve portfolio decisions.

### Key Features

- **Market data:** Uses `yfinance` to download Bitcoin prices.

- **Sentiment data:** Gets the Crypto Fear & Greed Index via the Alternative.me API.

- **Regime detection:** Fits a Markov regression model to find market states.

- **Dynamic allocation:** Uses the models regime probabilities to decide between risk-on and risk-off positions.

- **Backtesting:** Tests the strategy against a buy-and-hold approach.

- **Performance metrics:** Calculates annualized return, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown and cumulative return.

- **Visualization:** Creates a chart comparing strategy and benchmark performance.

- **Reproducibility:** Uses a Python environment and a dependency file.

## How It Works

The project follows these steps:

1. **Download market data**. Get Bitcoin prices for `BTC-USD`.

2. **Calculate returns**. Convert prices into log returns.

3. **Collect sentiment data**. Fetch the Fear & Greed Index from Alternative.me.

4. **Align the datasets**. Match sentiment with market dates. Handle missing values.

5. **Prepare model inputs**. Build lagged sentiment features for the model.

6. **Fit the model**. Estimate a Markov regression model with hidden states.

7. **Estimate market risk**. Use filtered regime probabilities to guide portfolio exposure.

8. ** The strategy**. Apply a one-day execution lag and include transaction costs.

9. **Evaluate performance**. Compare strategy results with buy-and-hold.

> **Modeling note:** In this version sentiment is used as a variable in the return equation. It does not affect how the Markov process switches between states. The strategy uses filtered probabilities for trades. Smoothed probabilities include data and are only for retrospective analysis.

## Technology Stack

- Python 3.12

- NumPy

- pandas

- yfinance

- statsmodels

- Matplotlib

- requests

- Alternative.me Fear & Greed Index API

## Project Structure

```

sentiment-markov-regime/

├── sentiment_regime_switching.py

├── requirements.txt

├── README.md

├──.gitignore

└─ regime_switching_performance.png  # Generated output; ignored by Git

```

The `.venv/` directory holds the local Python virtual environment. It is not tracked in version control.

## Installation

### 1. Clone the repository

```bash

git clone https://github.com/aabi-0111/sentiment-marko-regime.git

cd sentiment-marko-regime

```

### 2. Create an environment

For Windows:

```powershell

py -3.12 -m venv.venv

```

### 3. Activate the environment

```powershell

.\.venv\Scripts\Activate.ps1

```

### 4. Install dependencies

```powershell

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```

## Running the Project

Run the script:

```powershell

python -u sentiment_regime_switching.py

```

The script downloads data fits the model runs the backtest prints a performance report and generates a chart.

The output chart is saved as:

```text

regime_switching_performance.png

```

**Important:** If real sentiment data is not available the script uses sentiment. Check the console logs to see which sentiment source was used in the run.

## Performance Evaluation

The strategy is evaluated using these metrics:

| Metric | Purpose |

|---|---|

Annualized return | Measures the strategys annual growth rate |

| Annualized volatility | Measures how much returns fluctuate |

| Sharpe ratio | Measures return per unit of total risk |

Sortino ratio | Measures return per unit of downside risk |

| Maximum drawdown | Measures the drop from peak to trough |

| Cumulative return | Measures return over the time period |

Results are compared to a buy-and-hold Bitcoin benchmark. Performance varies based on the data period model estimates, allocation threshold and assumed transaction costs.

## Limitations and Research Considerations

- Past performance does not guarantee results.

- The model is trained on all the data. A proper out-of-sample or walk-forward test is needed for evaluation.

- Using full-sample standardization can lead to look-ahead bias.

- Smoothed regime probabilities include future data. They should not be used for trading signals.

- Sentiment data can have delays, gaps or revisions that affect results.

- The model includes transaction costs, but not spreads, market impact, taxes, funding costs or cash yield.

- The strategy may be sensitive to model assumptions and parameter choices.

This project is for research use only. It is not advice or a recommendation to trade cryptocurrency.

## Future Improvements

- Add walk-forward and out-of-sample backtesting.

- Use observed sentiment for valid research runs.

- Test whether sentiment affects transition probabilities.

- Evaluate the strategy on other assets and benchmarks.

- Run sensitivity analysis, on thresholds and transaction costs.

- Use realistic execution assumptions.

- Improve experiment. Reproducibility.

Author

Abhyansha Pandey

GitHub: [@aabi-0111](https://github.com/aabi-0111)

---

*Built as a finance research project exploring sentiment, latent market regimes and systematic portfolio risk management.*
