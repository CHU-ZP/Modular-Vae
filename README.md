# Modular VAE Demo

A clean PyTorch demo of Variational Autoencoders as modular probabilistic latent representation learning.

```text
x -> Encoder -> mu, logvar -> q_phi(z|x) -> z -> Decoder -> x_hat
                                |
                              p_psi(z)
```

The fixed part is the probabilistic interface. The replaceable parts are the encoder backbone, decoder backbone, prior, and ELBO configuration.

## Results

All experiments use MNIST.

| Experiment | Backbone | Prior | KL mode | Loss | Recon | KL |
| --- | --- | --- | --- | ---: | ---: | ---: |
| MLP VAE | MLP | Standard normal | Analytic | 100.05 | 79.98 | 20.07 |
| CNN VAE | CNN | Standard normal | Analytic | 95.98 | 75.31 | 20.67 |
| Beta-VAE | MLP | Standard normal | Analytic | 142.58 | 107.85 | 8.68 |
| Transformer VAE | Patch Transformer | Standard normal | Analytic | 96.04 | 73.92 | 22.13 |
| Flow-prior VAE | MLP | RealNVP-style flow | Monte Carlo | 95.75 | 75.99 | 19.76 |

| Experiment | Reconstructions | Prior samples |
| --- | --- | --- |
| MLP VAE | <img src="assets/figures/mlp_reconstructions.png" width="260" alt="MLP reconstructions"> | <img src="assets/figures/mlp_samples_from_prior.png" width="260" alt="MLP prior samples"> |
| CNN VAE | <img src="assets/figures/cnn_reconstructions.png" width="260" alt="CNN reconstructions"> | <img src="assets/figures/cnn_samples_from_prior.png" width="260" alt="CNN prior samples"> |
| Beta-VAE | <img src="assets/figures/beta_vae_reconstructions.png" width="260" alt="Beta-VAE reconstructions"> | <img src="assets/figures/beta_vae_samples_from_prior.png" width="260" alt="Beta-VAE prior samples"> |
| Transformer VAE | <img src="assets/figures/transformer_reconstructions.png" width="260" alt="Transformer reconstructions"> | <img src="assets/figures/transformer_samples_from_prior.png" width="260" alt="Transformer prior samples"> |
| Flow-prior VAE | <img src="assets/figures/flow_prior_reconstructions.png" width="260" alt="Flow-prior reconstructions"> | <img src="assets/figures/flow_prior_samples_from_prior.png" width="260" alt="Flow-prior prior samples"> |

Full metric YAML files live in [`assets/results`](assets/results).

## Probabilistic Model

A VAE does not encode an image into a deterministic vector. The encoder parameterizes an approximate posterior:

$$
q_\phi(z \mid x) = \mathcal{N}\left(z; \mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x))\right)
$$

Sampling is written with the reparameterization trick:

$$
\epsilon \sim \mathcal{N}(0, I), \qquad z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
$$

The training objective is the negative ELBO:

$$
\mathcal{L}(x) = \mathbb{E}_{q_\phi(z \mid x)}\left[-\log p_\theta(x \mid z)\right] + \beta\,\mathrm{KL}\left(q_\phi(z \mid x) \Vert p_\psi(z)\right)
$$

For the standard Gaussian prior,

$$
p(z) = \mathcal{N}(0, I)
$$

the diagonal Gaussian KL is analytic:

$$
\mathrm{KL}\left(q_\phi(z \mid x) \Vert \mathcal{N}(0,I)\right) = \frac{1}{2}\sum_j\left(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\right)
$$

For learned or nonstandard priors, the KL is estimated with one posterior sample:

$$
\mathrm{KL}\left(q_\phi(z \mid x) \Vert p_\psi(z)\right) \approx \log q_\phi(z \mid x) - \log p_\psi(z), \qquad z \sim q_\phi(z \mid x)
$$

The flow prior transforms a standard Gaussian base variable:

$$
u \sim \mathcal{N}(0,I), \qquad z = f_\psi(u)
$$

Its density is computed by change of variables:

$$
\log p_\psi(z) = \log p_0(f_\psi^{-1}(z)) + \log\left|\det\frac{\partial f_\psi^{-1}}{\partial z}\right|
$$

## VAE Variants

All variants keep the same probabilistic route:

```text
x -> q_phi(z|x) -> z -> p_theta(x|z)
```

They differ only in which module is swapped.

- **MLP VAE** is the baseline experiment: a fully connected encoder and decoder, a diagonal Gaussian posterior, a standard Gaussian prior, and analytic KL.
- **CNN VAE** keeps the same posterior, prior, and ELBO, but replaces the MLP backbone with convolutional encoder and decoder networks that better match image structure.
- **Beta-VAE** keeps the MLP architecture and standard Gaussian prior, but increases the KL weight beta. This makes the latent space more strongly regularized, usually trading reconstruction sharpness for a more constrained representation.
- **Transformer VAE** is not a different probabilistic model. It is a patch-based Transformer backbone behind the same posterior, sampling, prior, and ELBO interface.
- **Flow-prior VAE** keeps the Gaussian posterior but replaces the standard normal prior with a RealNVP-style flow prior. Because the prior is no longer a plain standard Gaussian, its KL term is estimated with `log q_phi(z|x) - log p_psi(z)`.

## What Is Modular

- Posterior: `DiagonalGaussian`
- Prior: `StandardNormalPrior` or `FlowPrior`
- Encoder backbone: MLP, CNN, or patch Transformer
- Decoder backbone: MLP, CNN, or patch Transformer
- Objective: standard VAE, beta-VAE, analytic KL, or Monte Carlo KL

The main `VAE` class only knows how to:

```text
encode(x) -> q_phi(z|x)
sample z
decode(z) -> x_hat
```

Everything else is configured from YAML.

## Quick Use

Install PyTorch for your machine, then install the remaining dependencies:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

Run all experiments:

```bash
bash scripts/run_all_experiments.sh
```

Run one experiment:

```bash
python -m vae.train --config configs/mnist_mlp_standard.yaml
```

Generate evaluation and figures:

```bash
python -m vae.evaluate --checkpoint outputs/mnist_mlp_standard/checkpoint.pt
python -m vae.visualize --checkpoint outputs/mnist_mlp_standard/checkpoint.pt
```

## Project Map

```text
configs/                  YAML experiment configs
vae/distributions.py       DiagonalGaussian posterior
vae/priors.py              Standard normal and flow priors
vae/flows.py               RealNVP-style affine coupling flow
vae/losses.py              ELBO / beta-VAE objective
vae/builders.py            Config-driven component builders
vae/models/                MLP, CNN, and Transformer backbones
vae/train.py               Training entry point
vae/evaluate.py            Test-set evaluation
vae/sample.py              Prior sampling
vae/visualize.py           Figures and latent interpolation
docs/                      Short conceptual notes
assets/                    Published figures and metrics
```

## Documentation

- [VAE overview](docs/vae_overview.md)
- [ELBO](docs/elbo.md)
- [Reparameterization](docs/reparameterization.md)
- [Flow prior](docs/flow_prior.md)

## License

MIT.
