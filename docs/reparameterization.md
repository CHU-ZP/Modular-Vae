# Reparameterization

The encoder outputs `mu` and `logvar`, which define:

```text
q_phi(z|x) = N(mu(x), diag(sigma^2(x)))
```

Direct sampling would block gradients through the random draw. The reparameterization trick rewrites the sample as:

```text
epsilon ~ N(0, I)
z = mu + sigma * epsilon
```

The randomness is isolated in `epsilon`, while `z` remains differentiable with respect to encoder outputs.
