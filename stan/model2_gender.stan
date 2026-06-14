data {
  int<lower=0> N;                 
  vector[N] y;                    
  vector[N] ESCS;                 
  vector[N] DISCLIM;              
  vector[N] PERSEVAGR;            
  vector[N] MATHEFF;              
  vector[N] female;               
}
parameters {
  real alpha;                     
  real b_ESCS;                    
  real b_DISCLIM;                 
  real b_PERSEVAGR;               
  real b_MATHEFF;                 
  real gamma;                     
  real<lower=0> sigma;            
}
model {
  alpha       ~ normal(500, 50);
  b_ESCS      ~ normal(0, 30);
  b_DISCLIM   ~ normal(0, 30);
  b_PERSEVAGR ~ normal(0, 30);
  b_MATHEFF   ~ normal(0, 30);
  gamma       ~ normal(0, 30);
  sigma       ~ exponential(0.01);

  y ~ normal(alpha + b_ESCS * ESCS
                   + b_DISCLIM * DISCLIM
                   + b_PERSEVAGR * PERSEVAGR
                   + b_MATHEFF * MATHEFF
                   + gamma * female,
             sigma);
}
generated quantities {
  vector[N] log_lik;              
  vector[N] y_rep;                
  for (i in 1:N) {
    real mu_i = alpha + b_ESCS * ESCS[i]
                      + b_DISCLIM * DISCLIM[i]
                      + b_PERSEVAGR * PERSEVAGR[i]
                      + b_MATHEFF * MATHEFF[i]
                      + gamma * female[i];
    log_lik[i] = normal_lpdf(y[i] | mu_i, sigma);
    y_rep[i]   = normal_rng(mu_i, sigma);
  }
}
