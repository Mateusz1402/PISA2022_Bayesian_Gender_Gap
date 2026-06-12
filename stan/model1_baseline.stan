data {
  int<lower=0> N;                 // number of students
  vector[N] y;                    // PISA mathematics score (point scale)
  vector[N] ESCS;                 // standardized socio-economic status
  vector[N] DISCLIM;              // standardized disciplinary climate
  vector[N] PERSEVAGR;            // standardized perseverance
}
parameters {
  real alpha;                     // intercept (mean score at average covariates)
  real b_ESCS;                    // effect of ESCS      (points per 1 sd)
  real b_DISCLIM;                 // effect of DISCLIM   (points per 1 sd)
  real b_PERSEVAGR;               // effect of PERSEVAGR (points per 1 sd)
  real<lower=0> sigma;            // residual standard deviation
}
model {
  alpha       ~ normal(500, 50);
  b_ESCS      ~ normal(0, 30);
  b_DISCLIM   ~ normal(0, 30);
  b_PERSEVAGR ~ normal(0, 30);
  sigma       ~ exponential(0.01);

  y ~ normal(alpha + b_ESCS * ESCS
                   + b_DISCLIM * DISCLIM
                   + b_PERSEVAGR * PERSEVAGR,
             sigma);
}
generated quantities {
  vector[N] log_lik;              // for WAIC / PSIS-LOO
  vector[N] y_rep;                // posterior predictive replicates
  for (i in 1:N) {
    real mu_i = alpha + b_ESCS * ESCS[i]
                      + b_DISCLIM * DISCLIM[i]
                      + b_PERSEVAGR * PERSEVAGR[i];
    log_lik[i] = normal_lpdf(y[i] | mu_i, sigma);
    y_rep[i]   = normal_rng(mu_i, sigma);
  }
}
