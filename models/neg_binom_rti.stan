data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  vector[N] gdp;        // log GDP per capita
  vector[N] log_pop;    // log population
}
parameters {
  real alpha;
  real beta;
  real<lower=0> phi;
}
model {
  // Priors
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  phi   ~ gamma(2, 0.1);
  
  // Likelihood: μ_i = exp(alpha + beta * gdp_i) * pop_i
  y ~ neg_binomial_2(exp(alpha + beta .* gdp) .* exp(log_pop), phi);
}
generated quantities {
  array[N] real log_lik;
  array[N] int y_rep;
  for (i in 1:N) {
    real mu = exp(alpha + beta * gdp[i]) * exp(log_pop[i]);
    log_lik[i] = neg_binomial_2_lpmf(y[i] | mu, phi);
    y_rep[i]   = neg_binomial_2_rng(mu, phi);
  }
}



