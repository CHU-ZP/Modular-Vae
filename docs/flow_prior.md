# Flow Prior

The standard VAE prior is:

```text
p(z) = N(0, I)
```

The flow prior starts from a standard Gaussian base variable and transforms it:

```text
u ~ N(0, I)
z = f_psi(u)
```

The log probability uses the inverse transformation:

```text
log p_psi(z) = log p0(u) + log |det df_psi^{-1} / dz|
```

This demo uses a compact RealNVP-style affine coupling flow. It is intentionally simple, meant to show how the prior can be replaced while the VAE interface stays fixed.
