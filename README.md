# investor-toolkit

Quantitative tools for long-term investors, built and tested like engineering simulation
software: every model is validated against an analytical or reference result.

> **Status:** under active development. Not investment advice.

## Modules

| Module | Question it answers |
|---|---|
| `data` | Where do the prices come from, and are they clean? |
| `returns` | What do returns look like: distribution, fat tails, horizon scaling? |
| `risk` | How bad can it get: volatility, VaR/CVaR, drawdowns? |
| `diversification` | Does combining assets reduce risk, and by how much? |
| `factors` | What drives an asset's returns? |
| `backtest` | How would a strategy have performed historically? |
| `montecarlo` | What range of outcomes can a savings plan expect? |

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/aperret04/investor-toolkit.git
cd investor-toolkit
uv sync
uv run pytest
```

## License

MIT
