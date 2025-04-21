# 1. Load libraries
library(cmdstanr)
library(posterior)
library(loo)
library(bayesplot)
library(MASS)
library(boot)
library(readr)
library(dplyr)
library(cowplot)
library(ggplot2)

# 2. Data preprocessing
traffic <- read_csv("data/clean/traffic_mortality_panel.csv") %>%
  filter(!is.na(gdp_pc), pop_total > 0, deaths >= 0) %>%
  mutate(
    log_gdp = log(gdp_pc),
    log_pop = log(pop_total)
  )

stan_data <- list(
  N       = nrow(traffic),
  y       = traffic$deaths,
  gdp     = traffic$log_gdp,
  log_pop = traffic$log_pop
)

# 3. Compile and sample the Stan model
mod <- cmdstan_model("models/neg_binom_rti.stan")
fit_bayes <- mod$sample(
  data          = stan_data,
  chains        = 4,
  iter_warmup   = 1000,
  iter_sampling = 1000,
  refresh       = 500
)

# 4. MCMC diagnostics
summ <- fit_bayes$summary(variables = c("alpha", "beta", "phi"))
cat("\n=== MCMC Convergence Diagnostics ===\n")
print(summ[, c("variable", "rhat", "ess_bulk", "ess_tail")])

# 5. Save trace & rank plots
trace_plot <- mcmc_trace(
  fit_bayes$draws(format = "draws_matrix", variables = c("alpha", "beta", "phi")),
  pars = c("alpha","beta","phi")
)
rank_plot <- mcmc_rank_hist(
  fit_bayes$draws(format = "draws_matrix", variables = c("alpha", "beta", "phi")),
  pars = c("alpha","beta","phi")
)
save_plot("report/trace_rank.png",
          plot       = plot_grid(trace_plot, rank_plot, ncol = 1),
          base_width = 6,
          base_height= 8)

# 6. Posterior predictive checks
y_rep <- fit_bayes$draws(format = "draws_matrix", variables = "y_rep")
ppc_density   <- ppc_dens_overlay(traffic$deaths, y_rep[1:200, ])
ppc_intervals <- ppc_intervals(traffic$deaths, y_rep[1:200, ])
save_plot("report/ppc_density.png",   ppc_density,   base_width = 6, base_height = 4)
save_plot("report/ppc_intervals.png", ppc_intervals, base_width = 6, base_height = 4)

ppc_dens_overlay(traffic$deaths, y_rep[1:200, ]) +
  ggtitle("PPC: Observed vs. Replicated Densities")

ppc_intervals(traffic$deaths, y_rep[1:200, ]) +
  ggtitle("PPC: 95% Prediction Intervals vs. Observed")

# 7. PSIS-LOO and WAIC
log_lik_matrix <- fit_bayes$draws(format = "draws_matrix", variables = "log_lik") %>%
  t()  # loo expects [S draws x N obs]
loo_bayes  <- loo(log_lik_matrix, reloo = TRUE)
waic_bayes <- waic(log_lik_matrix)
cat("\n=== Bayesian Model Comparison ===\n")
print(loo_bayes)
print(waic_bayes)

# 8. Frequentist NB with 5-fold CV
fit_freq <- glm.nb(deaths ~ log_gdp + offset(log_pop), data = traffic)
cv5 <- cv.glm(traffic, fit_freq, K = 5)
cat("\nFrequentist 5-fold CV error:", cv5$delta, "\n")

# 9. Decision theory: choose threshold to minimize expected loss
loss0 <- 10
loss1 <- 1
post_draws <- as_draws_matrix(y_rep)
thresholds <- seq(0, max(traffic$deaths)*1.5, by = 10)
expected_loss <- sapply(thresholds, function(T){
  losses <- ifelse(post_draws > T, loss1, loss0) 
  mean(losses)
})
best_T <- thresholds[which.min(expected_loss)]
cat("Optimal threshold:", best_T, "\n")