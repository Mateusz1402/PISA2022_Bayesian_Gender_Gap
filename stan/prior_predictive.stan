data {
  int<lower=0> N;                 
  vector[N] ESCS;                 
  vector[N] DISCLIM;              
  vector[N] PERSEVAGR;            
  vector[N] MATHEFF;              
}
generated quantities {
  real alpha       = normal_rng(500, 50);
  real b_ESCS      = normal_rng(0, 30);
  real b_DISCLIM   = normal_rng(0, 30);
  real b_PERSEVAGR = normal_rng(0, 30);
  real b_MATHEFF   = normal_rng(0, 30);
  real sigma       = exponential_rng(0.01);

  vector[N] y_sim;
  for (i in 1:N) {
    real mu_i = alpha + b_ESCS * ESCS[i]
                      + b_DISCLIM * DISCLIM[i]
                      + b_PERSEVAGR * PERSEVAGR[i]
                      + b_MATHEFF * MATHEFF[i];
    y_sim[i] = normal_rng(mu_i, sigma);
  }
}
