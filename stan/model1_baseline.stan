// =====================================================================
// Model 1 - BASELINE (no gender predictor)
//
//   y_i ~ Normal(mu_i, sigma)
//   mu_i = alpha + b_ESCS*ESCS_i + b_DISCLIM*DISCLIM_i
//                + b_PERSEVAGR*PERSEVAGR_i + b_MATHEFF*MATHEFF_i
//
// Priors (weakly-informative, on the PISA point scale):
//   alpha               ~ Normal(500, 50)
//   b_ESCS, b_DISCLIM,
//   b_PERSEVAGR,
//   b_MATHEFF           ~ Normal(0, 30)
//   sigma               ~ Exponential(0.01)
//
// MATHEFF (mathematics self-efficacy) is a strong, non-redundant control
// added to BOTH models so that the only structural difference between
// Model 1 and Model 2 remains the single gender parameter gamma.
//
// The generated quantities block produces:
//   * log_lik : pointwise log-likelihood, required for WAIC / PSIS-LOO
//   * y_rep   : posterior predictive replicates, for posterior predictive checks
// =====================================================================
data {
  int<lower=0> N;                 // number of students
  vector[N] y;                    // PISA mathematics score (point scale)
  vector[N] ESCS;                 // standardized socio-economic status
  vector[N] DISCLIM;              // standardized disciplinary climate
  vector[N] PERSEVAGR;            // standardized perseverance
  vector[N] MATHEFF;              // standardized mathematics self-efficacy
}
parameters {
  real alpha;                     // intercept (mean score at average covariates)
  real b_ESCS;                    // effect of ESCS      (points per 1 sd)
  real b_DISCLIM;                 // effect of DISCLIM   (points per 1 sd)
  real b_PERSEVAGR;               // effect of PERSEVAGR (points per 1 sd)
  real b_MATHEFF;                 // effect of MATHEFF   (points per 1 sd)
  real<lower=0> sigma;            // residual standard deviation
}
model {
  // ---- priors ----
  alpha       ~ normal(500, 50);
  b_ESCS      ~ normal(0, 30);
  b_DISCLIM   ~ normal(0, 30);
  b_PERSEVAGR ~ normal(0, 30);
  b_MATHEFF   ~ normal(0, 30);
  sigma       ~ exponential(0.01);

  // ---- likelihood (vectorized) ----
  y ~ normal(alpha + b_ESCS * ESCS
                   + b_DISCLIM * DISCLIM
                   + b_PERSEVAGR * PERSEVAGR
                   + b_MATHEFF * MATHEFF,
             sigma);
}
generated quantities {
  vector[N] log_lik;              // for WAIC / PSIS-LOO
  vector[N] y_rep;                // posterior predictive replicates
  for (i in 1:N) {
    real mu_i = alpha + b_ESCS * ESCS[i]
                      + b_DISCLIM * DISCLIM[i]
                      + b_PERSEVAGR * PERSEVAGR[i]
                      + b_MATHEFF * MATHEFF[i];
    log_lik[i] = normal_lpdf(y[i] | mu_i, sigma);
    y_rep[i]   = normal_rng(mu_i, sigma);
  }
}
