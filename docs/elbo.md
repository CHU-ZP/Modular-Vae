# ELBO

The training objective is the negative evidence lower bound:

```text
loss = reconstruction_loss + beta * KL(q_phi(z|x) || p_psi(z))
```

For the standard normal prior, the KL is analytic for a diagonal Gaussian posterior.

For arbitrary priors, this demo estimates the KL with one posterior sample:

```text
KL(q_phi(z|x) || p_psi(z)) ~= log q_phi(z|x) - log p_psi(z)
```

Changing `beta` gives beta-VAE behavior without changing the model class.
