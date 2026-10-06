# investor-toolkit: development guide

## Project

A Python package of quantitative tools for long-term investors. Each module answers one
question:

| Module | Question |
|---|---|
| `data` | Where do the prices come from, and are they clean? |
| `returns` | What do returns look like (distribution, tails, horizon scaling)? |
| `risk` | How bad can it get (volatility, VaR/CVaR, drawdowns)? |
| `diversification` | Does combining assets reduce risk, and by how much? |
| `factors` | What drives an asset's returns (factor exposures)? |
| `backtest` | How would a strategy have performed historically? |
| `montecarlo` | What range of outcomes can a savings plan expect? |

## Principles

- **Validate every model.** Each model has at least one test against an analytical
  solution or a reference result, marked `@pytest.mark.validation`. Monte Carlo methods
  include convergence checks.
- **Pure functions over state.** Functions take arrays or pandas objects and return
  arrays, pandas objects or small dataclasses. Plotting lives in `plotting.py`, never in
  model code.
- **No network in the default test suite.** Tests use synthetic data or small fixtures.
  Tests that need the internet are marked `@pytest.mark.network`.
- **Explicit conventions.** State units and conventions in docstrings: simple vs log
  returns, periodicity, annualisation factor, currency.
- **Written for external readers.** Docs, notebooks and comments are self-contained and
  professional.

## Layout

- `src/investor_toolkit/`: package source
- `tests/`: unit and validation tests
- `notebooks/`: worked examples that import from the package
- `data/`: local download cache (git-ignored)

## Commands

```
uv sync                         # install environment
uv run pytest                   # tests (offline)
uv run pytest -m network        # tests that download data
uv run ruff check . --fix       # lint
uv run ruff format .            # format
```

## Style

- Python 3.14, type hints, NumPy-style docstrings.
- Default currency EUR; default annualisation 252 trading days or 12 months.
- Random number generation through `numpy.random.Generator` with an explicit seed argument.
