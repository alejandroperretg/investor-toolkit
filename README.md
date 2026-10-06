# investor-toolkit

[![CI](https://github.com/aperret04/investor-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/aperret04/investor-toolkit/actions/workflows/ci.yml)

Quantitative tools for long-term investors in euros, built and tested like engineering simulation
software: every model is checked against a result it has to reproduce before it is used on
questions without a known answer.

> Not investment or tax advice. Results describe historical data (1999 to 2026) and model assumptions.

## At a glance

Six results, each as a concrete euro example. The full derivations are in the
[notebooks](notebooks); this section is reproduced by [notebook 0](notebooks/00_at_a_glance.ipynb).

### 1. One savings plan has many possible futures

Saving **500 a month for 30 years** (180,000 paid in) in a 60/40 portfolio, after costs and German
tax, ends at a **median of about 320,000** in today's money. But the range is wide: out of 100
simulated futures, the worst ends near 130,000 and the best above 750,000, and **6 end below the
amount paid in** once inflation is accounted for.

![100 futures of the same savings plan](docs/figures/glance_futures.png)

### 2. Crashes are deep, and recovery can take a decade

**10,000 invested in world equities at the August 2000 peak was worth 4,751 in March 2003**. It
took until February 2013 to get back to 10,000, and it had grown to about 50,600 by 2026. Whoever
had to sell in 2003 lost half; whoever could wait was rewarded.

![10,000 invested at the 2000 peak](docs/figures/glance_crash.png)

### 3. Extreme days happen far more often than the bell curve says

Daily moves of the MSCI World ETF since 2010, compared with a normal-distribution model:

| Move larger than | Days it happened (17 years) | Normal model: once every |
|---|---|---|
| 3 typical days (3 sigma) | 72 | 1.5 years |
| 4 typical days (4 sigma) | 24 | 63 years |
| 5 typical days (5 sigma) | 10 | about 7,000 years |

Risk models built on the normal distribution underestimate crashes. In an out-of-sample test, a
normal-model "1-in-100 months" loss limit was exceeded in 3.7 % of months instead of 1 %.

### 4. Bonds protect against recessions, not against inflation

In 2002 and 2008 German government bonds **rose while stocks fell 30 % and 37 %**. In 2022, when
inflation pushed interest rates up, **both fell together**: stocks 13 %, bonds 22 %.

![Calendar-year returns of stocks and bonds](docs/figures/glance_bonds.png)

### 5. Spreading risk works, up to a point

A single German blue-chip stock typically swings about 26 % in a year. Five of them together
swing about 20 %, because they still share one economy. A world index swings about 12 %.

![Volatility of one stock, five stocks and the world index](docs/figures/glance_diversification.png)

### 6. The 4 % rule is optimistic for euro investors

Retire with **500,000** and withdraw a fixed amount, raised with inflation every year. Over 30 years
of simulated euro-area history, **withdrawing 20,000 a year (4 %) ran out of money in 16 of 100
histories; 15,000 a year (3 %) in 3 of 100.**

![Withdrawal outcomes in 100 histories](docs/figures/glance_withdrawals.png)

**And a warning about backtests:** a trend-following rule turned 1 euro into 8.0 since 2002, against
6.8 for buy-and-hold, with smaller crashes. But when the history is reshuffled 2,000 times, it
comes out behind in about 9 of 100 cases, so the data cannot rule out luck
([notebook 6](notebooks/06_backtest.ipynb)).

### Terms in one line each

| Term | Meaning |
|---|---|
| Volatility | Typical yearly swing. 12 % means that in about two years out of three, the return lands within 12 points of its average. |
| Drawdown | Fall from the previous high. 10,000 falling to 4,750 is a 52 % drawdown. |
| Correlation | Whether two assets move together: +1 always together, 0 unrelated, -1 always opposite. |
| Sigma | The typical size of a move. A 5-sigma day is five times larger than a typical day. |
| Monte Carlo | Replaying thousands of possible market histories to see the range of outcomes. |
| Today's money | Amounts divided by cumulative inflation, so they buy what they would buy today. |

## How the models are checked

A simulation is only trusted after it reproduces cases with a known answer, the same way a
structural solver is first tested on a beam with a textbook solution. Examples:

| Check | Known answer | Code |
|---|---|---|
| Mean final wealth of a savings plan under geometric Brownian motion | 51,041 | 50,993 |
| Simulation error with 4 times more histories | halves (error ~ 1/√N) | slope -0.49 vs -0.50 |
| Expected maximum drawdown of Brownian motion, √(π/2) | 1.253 | 1.247 |
| Lifetime German fund tax = flat tax on the total gain | exact | equal to 10 digits |
| Ledoit-Wolf shrinkage, OLS and Newey-West standard errors | scikit-learn, statsmodels | equal to 8-10 digits |
| Synthetic Bund returns vs German government bond ETFs | correlation, ordered by duration | 0.95; betas 0.46 / 0.74 / 1.60 |

![Monte Carlo convergence](docs/figures/mc_convergence.png)

The full inventory is in [notebook 8](notebooks/08_validation.ipynb); all checks run in CI on
every push.

## Modules

Each module answers one investor question:

| Module | Question | Notebook |
|---|---|---|
| | The results as euro examples | [00 At a glance](notebooks/00_at_a_glance.ipynb) |
| `data`, `proxies` | Where do the numbers come from, and what happened before ETFs existed? | [01 Data](notebooks/01_data.ipynb) |
| `returns` | What do returns look like? | [02 Returns](notebooks/02_returns.ipynb) |
| `risk` | How bad can it get, and for how long? | [03 Risk](notebooks/03_risk.ipynb) |
| `diversification` | Does combining assets reduce risk, and by how much? | [04 Diversification](notebooks/04_diversification.ipynb) |
| `factors` | What drives an asset's returns? | [05 Factors](notebooks/05_factors.ipynb) |
| `backtest` | Would a strategy have worked, or was it luck? | [06 Backtest](notebooks/06_backtest.ipynb) |
| `montecarlo`, `tax` | What range of outcomes can a savings plan expect, and how much can be withdrawn? | [07 Monte Carlo](notebooks/07_montecarlo.ipynb) |
| | How do we know the models are right? | [08 Validation](notebooks/08_validation.ipynb) |

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/aperret04/investor-toolkit.git
cd investor-toolkit
uv sync
uv run pytest                  # offline unit and validation tests
uv run pytest -m network       # tests that download market data
```

Project a savings plan from bootstrapped history:

```python
from investor_toolkit import montecarlo as mc
from investor_toolkit import proxies
from investor_toolkit.tax import MIXED_FUND, GermanTax

history = proxies.long_history()  # downloads and caches the data on first use
balanced = 0.6 * history["world_equity"] + 0.4 * history["bund_10y"]
model = mc.Bootstrap(balanced, history["inflation"], mean_block=12)
plan = mc.SavingsPlan(
    monthly_contribution=500,
    accumulation_years=30,
    annual_fee=0.002,
    tax=GermanTax(partial_exemption=MIXED_FUND),
)
result = mc.simulate(model, plan, n_paths=10_000)
print(result.summary())
print(result.percentiles())  # wealth percentiles per year, in today's money
```

To run the notebooks, open them in VS Code (or any Jupyter front end) and select the project's
`.venv` as the kernel.

## Data

| Source | Series |
|---|---|
| [Yahoo Finance](https://finance.yahoo.com) via `yfinance` | Xetra-listed UCITS ETFs and stocks (total-return prices) |
| [FRED](https://fred.stlouisfed.org) | USD/EUR (`DEXUSEU`), German 3-month rate, euro-area HICP |
| [Deutsche Bundesbank](https://www.bundesbank.de/en/statistics) | Daily 10-year German government yield |
| [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Developed and European factor returns |

European ETFs only exist since about 2009. The history before that is built from official index
data (see [notebook 1](notebooks/01_data.ipynb)). Data are downloaded on first use and cached in
`data/` (not part of the repository).

## Project layout

```
src/investor_toolkit/
    data.py             cached loaders, currency conversion, quality checks
    proxies.py          long-history EUR series (equity, synthetic Bund, cash, inflation)
    universe.py         assets and model portfolios
    returns.py          return conventions, distribution fits, tails, variance ratios
    risk.py             VaR/ES, VaR backtests, drawdowns, risk-adjusted ratios
    diversification.py  covariance shrinkage, frontier, risk parity, conditional correlation
    factors.py          Fama-French regressions with robust standard errors
    backtest.py         look-ahead-free engine, strategies, deflated Sharpe ratio
    resampling.py       stationary block bootstrap
    montecarlo.py       savings and withdrawal simulation
    tax.py              German investment-fund taxation
    plotting.py         figure style
tests/                  unit and validation tests
notebooks/              worked examples (generate docs/figures)
docs/methodology.md     equations, assumptions, limitations, references
```

## Methodology and limitations

Equations, conventions, assumptions and references are in
[docs/methodology.md](docs/methodology.md). The main limitation is sample length: about 27 years
of monthly data contain only a few independent bear markets, so long-horizon numbers carry wide
uncertainty. The tax model is a planning simplification.

Project page: [aperret04.github.io/work/investor-toolkit.html](https://aperret04.github.io/work/investor-toolkit.html)

## License

[MIT](LICENSE)
