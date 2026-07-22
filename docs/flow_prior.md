# Flow Prior

The standard VAE prior is:

$$
p(z) = \mathcal{N}(0, I)
$$

The flow prior starts from a standard Gaussian base variable and transforms it:

$$
u \sim \mathcal{N}(0, I)
$$

$$
z = f_\psi(u)
$$

The log probability uses the inverse transformation. First map the observed latent point back to the base variable:

$$
u = f_\psi^{-1}(z)
$$

Then apply the change-of-variables formula:

$$
\log p_\psi(z) = \log p_0(u) + \log \lvert \det(\partial u / \partial z) \rvert
$$

This demo uses a compact RealNVP-style affine coupling flow. It is intentionally simple, meant to show how the prior can be replaced while the VAE interface stays fixed.
