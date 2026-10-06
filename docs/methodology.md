# Methodology

This document summarises the models, conventions and assumptions behind each module. Worked
examples are in the [notebooks](../notebooks); validation results are in
[notebook 8](../notebooks/08_validation.ipynb) and the test suite.

## Conventions

| Item | Convention |
|---|---|
| Currency | EUR. Foreign-currency series are converted with month-end (or daily) exchange rates. |
| Returns | Simple returns $R_t = P_t / P_{t-1} - 1$ for portfolios; log returns $r_t = \ln(1 + R_t)$ for time aggregation and distribution fitting. |
| Frequency | Daily (252 periods per year) for distribution analysis; monthly (12) for portfolios, backtests and projections. |
| Prices | Total-return prices: adjusted for splits and distributions, net of fund costs. |
| Risk measures | VaR and ES are positive numbers representing losses. |
| Random numbers | `numpy.random.Generator` with explicit seeds. |

## Data

**Investable universe.** Xetra-listed UCITS ETFs and five large German stocks (Yahoo Finance).
Prices are cleaned by forward-filling gaps of at most five trading days within each series' live
range. Monthly returns run month end to month end; an incomplete final month is dropped.

**Long-history proxies (1999 onwards).**

* *World equity:* Fama-French developed-market return, $R^{USD} = \text{Mkt-RF} + \text{RF}$,
  converted to EUR as $(1 + R^{USD})\,X_{t-1}/X_t - 1$ with $X$ in USD per EUR (FRED `DEXUSEU`).
* *10-year Bund:* constant-maturity par bond priced from the Bundesbank daily 10-year yield
  (month-end values). With coupon $c = y_{t-1}$, maturity $M = 10$ and $\Delta = 1/12$:

  $$R_t = c\,a(y_t, M - \Delta) + (1 + y_t)^{-(M-\Delta)} - 1 + c\Delta,\qquad
  a(y, T) = \frac{1 - (1+y)^{-T}}{y}$$

  The modified duration of the bond is $a(y, M)$, so
  $R_t \approx y_{t-1}\Delta - a\,(y_t - y_{t-1})$. The Svensson spot rate is used as an
  approximation of the par yield.
* *Cash:* German 3-month interbank rate (monthly average), earned over the following month.
* *Inflation:* euro-area HICP.

The equity proxy is validated against the MSCI World ETF and the bond proxy against German
government bond ETFs over the overlapping period (correlation, tracking error, beta).

## Return distributions (`returns`)

* Normal and Student-t distributions fitted by maximum likelihood, compared by AIC, BIC and
  the Kolmogorov-Smirnov statistic.
* Tail frequencies beyond $k$ standard deviations vs the normal prediction $2n\,\Phi(-k)$.
* Hill (1975) tail index from the $k$ largest losses or gains:
  $\hat\alpha^{-1} = \frac{1}{k}\sum_{i=1}^{k} \ln X_{(i)} - \ln X_{(k+1)}$.
* Lo and MacKinlay (1988) variance-ratio test with overlapping observations and the
  heteroskedasticity-robust statistic $z^*(q)$. For AR(1) returns, $VR(2) = 1 + \rho_1$.

## Risk (`risk`)

* **VaR** at tail probability $\alpha$: historical quantile; Gaussian $-(\mu + \sigma z_\alpha)$;
  Cornish-Fisher expansion with sample skewness $S$ and excess kurtosis $K$,
  $z_{cf} = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36$; Student-t quantile.
* **Expected shortfall:** historical tail mean; analytical Gaussian
  $-\mu + \sigma\,\varphi(z_\alpha)/\alpha$; analytical Student-t
  $-\mu + s\,g_\nu(q)\,(\nu + q^2)/((\nu-1)\alpha)$ with $q = t_\nu^{-1}(1-\alpha)$
  (McNeil, Frey and Embrechts, 2015).
* **VaR backtesting:** rolling out-of-sample forecasts; Kupiec (1995) proportion-of-failures
  and Christoffersen (1998) independence likelihood-ratio tests.
* **Drawdowns:** $D_t = W_t / \max_{s \le t} W_s - 1$, with episodes (peak, trough, recovery) and
  the ulcer index $\sqrt{\overline{D^2}}$. Reference: the expected maximum drawdown of driftless
  Brownian motion is $\sqrt{\pi/2}\,\sigma\sqrt{T}$ (Magdon-Ismail et al., 2004).
* **Ratios:** Sharpe, Sortino (downside deviation of excess returns), Calmar (CAGR over maximum
  drawdown).

## Diversification (`diversification`)

* Equal-weight variance with common volatility and correlation:
  $\sigma_p^2 = \sigma^2(\rho + (1-\rho)/N)$.
* Risk contributions $w_i(\Sigma w)_i / w^\top\Sigma w$ and the diversification ratio
  $w^\top\sigma / \sqrt{w^\top\Sigma w}$.
* Ledoit and Wolf (2004) shrinkage towards a scaled identity with the optimal intensity.
* Minimum-variance, tangency and efficient-frontier portfolios: closed forms without constraints
  (Merton, 1972), SLSQP with long-only constraints.
* Risk parity (equal or budgeted risk contributions) by cyclical coordinate descent on Spinu's
  (2013) convex formulation (Griveau-Billion, Richard and Roncalli, 2013).
* Down-market correlation compared with the bivariate-normal benchmark
  $\rho_A = \rho/\sqrt{\rho^2 + (1-\rho^2)/v_A}$, where $v_A$ is the variance of the truncated
  standard normal (Boyer, Gibson and Loretan, 1997).

## Factor models (`factors`)

* Fama and French (2015) five factors plus momentum (Carhart, 1997) from the Kenneth R. French
  Data Library, in USD; asset returns are converted to USD before regressing.
* Ordinary least squares with classical, White (HC0) or Newey and West (1987) standard errors,
  Bartlett kernel, lag length $\lfloor 4(n/100)^{2/9}\rfloor$.
* Return attribution: mean excess return $= \alpha + \sum_k \beta_k \bar f_k$ (exact with an
  intercept).

## Backtesting (`backtest`)

* Monthly simulation. Targets for month $t$ use data up to $t-1$ only; trades occur at the start of
  the month; costs are proportional to traded value.
* Rebalancing on a calendar, on a tolerance band, or never; contributions are invested at target
  weights.
* Time-weighted returns (strategy) and money-weighted IRR (investor).
* Strategies: fixed mix, inverse volatility, risk parity and minimum variance (Ledoit-Wolf
  covariance, 36-month window), and a 10-month moving-average trend rule (Faber, 2007).
* Statistical evaluation: Sharpe ratio standard error
  $\sqrt{(1 - \gamma_3 SR + \frac{\gamma_4 - 1}{4} SR^2)/(n-1)}$ (Lo, 2002; Mertens, 2002);
  probabilistic Sharpe ratio (Bailey and Lopez de Prado, 2012); deflated Sharpe ratio with the
  expected maximum of $N$ null Sharpe ratios
  $\sqrt{V}\,[(1-\gamma)\Phi^{-1}(1 - 1/N) + \gamma\Phi^{-1}(1 - 1/(Ne))]$
  (Bailey and Lopez de Prado, 2014); paired stationary-bootstrap confidence intervals.

## Resampling (`resampling`)

Stationary bootstrap (Politis and Romano, 1994): blocks of geometrically distributed length with
mean $L$, starting at uniform random positions, wrapping circularly. Conditional on the data, the
bootstrap variance of the mean is $\frac{1}{n}[\hat\gamma_0 + 2\sum_k (1 - k/n)(1 - 1/L)^k\hat\gamma_k]$
with circular sample autocovariances.

## Monte Carlo projections (`montecarlo`, `tax`)

* **Return models.** GBM: monthly log returns $\mathcal N(a, b^2)$ with $b = \sigma/\sqrt{12}$ and
  $a = \ln(1 + \mu)/12 - b^2/2$, so that $E[1 + R_{year}] = 1 + \mu$. Student-t: the same mean and
  variance of log returns with unit-variance t shocks. Block bootstrap: historical months of
  portfolio return and inflation resampled jointly.
* **Exact results under GBM** used for validation: with contributions $c_k$ and iid monthly growth
  factors ($E[G] = g$, $E[G^2] = h$),
  $E[W] = \sum_k c_k g^{n-k}$ and $E[W^2] = \sum_{k,l} c_k c_l\, g^{|k-l|} h^{n - \max(k,l)}$.
* **Plan.** Monthly contributions (optionally indexed to inflation) followed by inflation-indexed
  withdrawals, either a fixed amount or a fraction of wealth at the start of the withdrawal phase.
  A path fails if a withdrawal cannot be paid in full.
* **German taxation** of an accumulating fund (Investment Tax Act, InvStG 2018): flat tax 26.375 %
  (25 % plus solidarity surcharge, no church tax); partial exemption 30 % / 15 % / 0 % for equity /
  mixed / other funds; saver's allowance EUR 1,000 per year; annual Vorabpauschale
  $\min(0.7 \cdot \text{base rate} \cdot P_{start},\ \max(P_{end} - P_{start}, 0))$ per unit, reduced
  by 1/12 per full month before purchase; Vorabpauschalen credited against gains on sale; FIFO with
  one lot per purchase year; losses carried forward. Taxes are paid from the portfolio by default.
* **Safe withdrawal rate:** root search on the success rate using common random numbers.

## Limitations

* **Sample length.** About 27 years of monthly data contain few independent bear markets. Long
  horizon statistics (multi-year volatility scaling, strategy comparisons, safe withdrawal rates)
  carry wide uncertainty.
* **Proxies.** Index proxies carry no fund costs; the bond proxy approximates the par yield with a
  spot rate; the cash rate is a monthly average.
* **Selection.** The five single stocks were chosen with knowledge of the present; any named stock
  basket is subject to selection and survivorship bias.
* **Tax model.** A planning simplification: portfolio-level partial exemption for mixed
  allocations, annual FIFO lots, no church tax, no personal tax-rate option
  (Günstigerprüfung), and a constant base rate (Basiszins). Not tax advice.
* **Costs.** Proportional transaction costs only; no bid-ask spread modelling, taxes in backtests,
  or market impact.

## References

* Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2).
* Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5).
* Boyer, B. H., Gibson, M. S. and Loretan, M. (1997). Pitfalls in tests for changes in correlations. *Federal Reserve International Finance Discussion Paper* 597.
* Broadie, M., Glasserman, P. and Kou, S. (1997). A continuity correction for discrete barrier options. *Mathematical Finance*, 7(4).
* Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1).
* Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4).
* Faber, M. T. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4).
* Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1).
* Griveau-Billion, T., Richard, J.-C. and Roncalli, T. (2013). A fast algorithm for computing high-dimensional risk parity portfolios. SSRN.
* Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution. *Annals of Statistics*, 3(5).
* Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2).
* Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2).
* Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4).
* Lo, A. W. and MacKinlay, A. C. (1988). Stock market prices do not follow random walks. *Review of Financial Studies*, 1(1).
* Magdon-Ismail, M., Atiya, A. F., Pratap, A. and Abu-Mostafa, Y. S. (2004). On the maximum drawdown of a Brownian motion. *Journal of Applied Probability*, 41(1).
* McNeil, A. J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk Management*, revised edition. Princeton University Press.
* Merton, R. C. (1972). An analytic derivation of the efficient portfolio frontier. *Journal of Financial and Quantitative Analysis*, 7(4).
* Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3).
* Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4).
* Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428).
* Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN.
