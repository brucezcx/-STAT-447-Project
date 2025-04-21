# STAT-447-Project
# Bayesian vs. Frequentist Negative Binomial Regression

This project compares Bayesian and frequentist approaches to modeling road traffic mortality using Negative Binomial regression.

## Goal

Understand how GDP per capita affects traffic mortality, controlling for population size.

## Model

\[
y_i \sim \text{NegBin}(\mu_i, \phi), \quad \log(\mu_i) = \alpha + \beta \cdot \log(\text{gdp}_i) + \log(\text{pop}_i)
\]

## Methods

- **Frequentist**: `glm.nb()` (MASS)
- **Bayesian**: Stan + MCMC


## Files

- `EDA.Rmd`: Data exploration
- `model_frequentist.R`: Frequentist model
- `model_bayesian.stan`: Bayesian model
- `report.Rmd`: Full analysis

## Author

Chuxuan Zhou ([GitHub](https://github.com/brucezcx))
