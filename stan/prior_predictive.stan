// =====================================================================
// PRIOR PREDICTIVE model (baseline structure, no gender term)
//
// This program contains NO likelihood. It draws parameters from their
// priors and pushes them through the generative model to simulate
// measurements y_sim. It is meant to be run with the "fixed_param"
// sampler (algorithm=fixed_param), which simply evaluates the
// generated quantities block once per iteration.
//
// The baseline structure now includes the MATHEFF control (mathematics
// self-efficacy), matching the updated Model 1 / Model 2.
//
// Used in the notebook for the prior predictive check on MEASUREMENTS:
// do data simulated purely from the priors look like plausible PISA
// scores, BEFORE we condition on the observed outcomes?
// =====================================================================
data {
  int<lower=0> N;                 // number of students
  vector[N] ESCS;                 // standardized socio-economic status
  vector[N] DISCLIM;              // standardized disciplinary climate
  vector[N] PERSEVAGR;            // standardized perseverance
  vector[N] MATHEFF;              // standardized mathematics self-efficacy
}
generated quantities {
  // ---- draw parameters from their PRIORS ----
  real alpha       = normal_rng(500, 50);
  real b_ESCS      = normal_rng(0, 30);
  real b_DISCLIM   = normal_rng(0, 30);
  real b_PERSEVAGR = normal_rng(0, 30);
  real b_MATHEFF   = normal_rng(0, 30);
  real sigma       = exponential_rng(0.01);

  // ---- simulate measurements from the generative model ----
  vector[N] y_sim;
  for (i in 1:N) {
    real mu_i = alpha + b_ESCS * ESCS[i]
                      + b_DISCLIM * DISCLIM[i]
                      + b_PERSEVAGR * PERSEVAGR[i]
                      + b_MATHEFF * MATHEFF[i];
    y_sim[i] = normal_rng(mu_i, sigma);
  }
}
