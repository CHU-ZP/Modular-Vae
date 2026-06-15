# VAE Overview

A Variational Autoencoder learns a latent-variable model with two distributions:

```text
q_phi(z|x)  approximate posterior from data to latent
p_theta(x|z) decoder / likelihood from latent to data
```

This demo keeps that probabilistic interface fixed:

```text
x -> q_phi(z|x) -> z -> p_theta(x|z)
```

MLP, CNN, and Transformer modules are only replaceable backbones used to parameterize the same distributions.
