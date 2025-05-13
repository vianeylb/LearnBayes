data {
  int<lower = 0> N;  // number of states
  int<lower = 1> T;  // number of observations
  array[T] real y;
}

parameters {
  array[N] simplex[N] theta;  // N x N tp
  vector[N] mu;  // state-dependent parameters
  vector<lower=0.0>[N] sigma;
  simplex[N] init_dist;
}


transformed parameters {
  
  matrix[N, T] log_omega;
  matrix[N, N] Gamma;

  // build log_omega
  for (t in 1:T)
    for (n in 1:N) log_omega[n,t] = normal_lpdf(y[t] | mu[n], sigma[n]);

  // build Gamma
  for (n in 1:N) Gamma[n, ] = theta[n]';

}

model {
  
  // priors
  mu ~ normal(2, 3);
  sigma ~ normal(0.5, 1);  
  
  target += hmm_marginal(log_omega, Gamma, init_dist);
  
}

generated quantities{

  matrix[N, T] states_prob = hmm_hidden_state_prob(log_omega, Gamma, init_dist);
  array[T] int states_pred = hmm_latent_rng(log_omega, Gamma, init_dist);

   vector[T] y_pred;
   for (i in 1:T)
     y_pred[i] = normal_rng(mu[states_pred[i]], sigma[states_pred[i]]);

}

