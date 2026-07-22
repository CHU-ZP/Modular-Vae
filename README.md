# Modular VAE

<p align="right">
English | <a href="README.zh-CN.md">中文</a>
</p>

A compact VAE playground that maps probabilistic latent representation
learning to small, readable PyTorch modules with replaceable backbones and
priors.

```text
image -> encoder -> posterior q(z|x) -> latent z -> decoder -> reconstruction
                              |
                           prior p(z)
```

The repository is built as a learning-oriented implementation: the
probabilistic interface stays fixed while the encoder, decoder, prior, and ELBO
configuration remain explicit experimental choices.

## The Idea

A Variational Autoencoder does not map an image to one deterministic vector.
The encoder parameterizes an approximate posterior distribution:

```math
q_\phi(z \mid x)
=
\mathcal{N}\left(
z;
\mu_\phi(x),
\mathrm{diag}(\sigma_\phi^2(x))
\right).
```

Sampling uses the reparameterization trick so gradients can flow through the
posterior parameters:

```math
\epsilon \sim \mathcal{N}(0,I),
\qquad
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon.
```

The decoder maps the sampled latent back to an image:

```math
z \sim q_\phi(z\mid x),
\qquad
\hat{x}=\mathrm{Decoder}_\theta(z).
```

The shared probabilistic wrapper is implemented in
[`vae/models/base_vae.py`](vae/models/base_vae.py), and the diagonal posterior
is implemented in [`vae/distributions.py`](vae/distributions.py).

## The Training Objective

Training minimizes the negative evidence lower bound: a reconstruction term
plus a KL regularizer that aligns the approximate posterior with the prior.

```math
\mathcal{L}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[-\log p_\theta(x\mid z)\right]
+
\beta\,
\mathrm{KL}\left(q_\phi(z\mid x)\Vert p_\psi(z)\right).
```

For the standard Gaussian prior, the diagonal Gaussian KL is analytic:

```math
\mathrm{KL}\left(q_\phi(z\mid x)\Vert\mathcal{N}(0,I)\right)
=
\frac{1}{2}
\sum_j
\left(
\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2
\right).
```

For learned or nonstandard priors, the KL is estimated with one posterior
sample:

```math
\mathrm{KL}\left(q_\phi(z\mid x)\Vert p_\psi(z)\right)
\approx
\log q_\phi(z\mid x)-\log p_\psi(z),
\qquad
z\sim q_\phi(z\mid x).
```

The objective is implemented in [`vae/losses.py`](vae/losses.py).

## Sampling

Generation does not require an input image. A latent is drawn from the prior
and passed directly through the decoder:

```math
z\sim p_\psi(z),
\qquad
x\sim p_\theta(x\mid z).
```

The standard experiment uses a fixed Gaussian prior:

```math
p(z)=\mathcal{N}(0,I).
```

The flow-prior experiment transforms a standard Gaussian base variable with a
RealNVP-style affine coupling flow:

```math
u\sim\mathcal{N}(0,I),
\qquad
z=f_\psi(u).
```

Its density is evaluated through the inverse transformation:

```math
u=f_\psi^{-1}(z),
\qquad
\log p_\psi(z)
=
\log p_0(u)
+
\log\left|\det\frac{\partial u}{\partial z}\right|.
```

Prior implementations live in [`vae/priors.py`](vae/priors.py), and the flow is
implemented in [`vae/flows.py`](vae/flows.py).

## Inside This Demo

All experiments use MNIST and keep the same posterior-sample-decoder route.
They change only the component needed for a controlled comparison.

| Experiment | Backbone | Prior | Objective | Purpose |
| --- | --- | --- | --- | --- |
| MLP VAE | MLP | Standard normal | Analytic KL, `beta=1` | Fully connected baseline |
| CNN VAE | CNN | Standard normal | Analytic KL, `beta=1` | Image-structure inductive bias |
| Beta-VAE | MLP | Standard normal | Analytic KL, `beta=4` | Stronger latent regularization |
| Transformer VAE | Patch Transformer | Standard normal | Analytic KL, `beta=1` | Patch-based attention backbone |
| Flow-prior VAE | MLP | RealNVP-style flow | Monte Carlo KL, `beta=1` | Learned non-Gaussian prior |

The modular boundaries are:

| Component | Implementation |
| --- | --- |
| Diagonal Gaussian posterior | [`vae/distributions.py`](vae/distributions.py) |
| Standard and flow priors | [`vae/priors.py`](vae/priors.py) |
| RealNVP-style flow | [`vae/flows.py`](vae/flows.py) |
| MLP, CNN, and Transformer encoders | [`vae/models/encoders.py`](vae/models/encoders.py) |
| MLP, CNN, and Transformer decoders | [`vae/models/decoders.py`](vae/models/decoders.py) |
| ELBO and beta-VAE objective | [`vae/losses.py`](vae/losses.py) |
| Configuration-driven builders | [`vae/builders.py`](vae/builders.py) |

## Relation to Nearby Methods

| Method | Learned object | Generation |
| --- | --- | --- |
| VAE | approximate posterior, prior, and decoder | sample a latent and decode it |
| Diffusion | noise, score, data, or velocity prediction | reverse SDE, ODE, or denoising chain |
| Flow Matching | continuous vector field | solve an ODE from noise to data |
| This repository | replaceable VAE backbones, priors, and ELBO settings | latent sampling and MNIST decoding |

The VAE performs generation in one decoder pass after sampling a compact
latent. Diffusion and Flow Matching instead learn iterative dynamics in pixel
or latent space.

## Results

The table reports test-set loss components for the five completed MNIST runs.

| Experiment | Backbone | Prior | KL mode | Loss | Recon | KL |
| --- | --- | --- | --- | ---: | ---: | ---: |
| MLP VAE | MLP | Standard normal | Analytic | 100.05 | 79.98 | 20.07 |
| CNN VAE | CNN | Standard normal | Analytic | 95.98 | 75.31 | 20.67 |
| Beta-VAE | MLP | Standard normal | Analytic | 142.58 | 107.85 | 8.68 |
| Transformer VAE | Patch Transformer | Standard normal | Analytic | 96.04 | 73.92 | 22.13 |
| Flow-prior VAE | MLP | RealNVP-style flow | Monte Carlo | 95.75 | 75.99 | 19.76 |

| Experiment | Reconstructions | Prior samples |
| --- | --- | --- |
| MLP VAE | <img src="assets/figures/mlp_reconstructions.png" width="260" alt="MLP VAE reconstructions"> | <img src="assets/figures/mlp_samples_from_prior.png" width="260" alt="MLP VAE prior samples"> |
| CNN VAE | <img src="assets/figures/cnn_reconstructions.png" width="260" alt="CNN VAE reconstructions"> | <img src="assets/figures/cnn_samples_from_prior.png" width="260" alt="CNN VAE prior samples"> |
| Beta-VAE | <img src="assets/figures/beta_vae_reconstructions.png" width="260" alt="Beta-VAE reconstructions"> | <img src="assets/figures/beta_vae_samples_from_prior.png" width="260" alt="Beta-VAE prior samples"> |
| Transformer VAE | <img src="assets/figures/transformer_reconstructions.png" width="260" alt="Transformer VAE reconstructions"> | <img src="assets/figures/transformer_samples_from_prior.png" width="260" alt="Transformer VAE prior samples"> |
| Flow-prior VAE | <img src="assets/figures/flow_prior_reconstructions.png" width="260" alt="Flow-prior VAE reconstructions"> | <img src="assets/figures/flow_prior_samples_from_prior.png" width="260" alt="Flow-prior VAE prior samples"> |

Full metric YAML files live in [`assets/results/`](assets/results/).

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

Generate evaluation metrics and figures:

```bash
python -m vae.evaluate --checkpoint outputs/mnist_mlp_standard/checkpoint.pt
python -m vae.visualize --checkpoint outputs/mnist_mlp_standard/checkpoint.pt
```

## Project Map

```text
configs/                    YAML experiment configs
vae/distributions.py        Diagonal Gaussian posterior
vae/priors.py               Standard normal and flow priors
vae/flows.py                RealNVP-style affine coupling flow
vae/losses.py               ELBO and beta-VAE objective
vae/builders.py             Config-driven component builders
vae/models/                 MLP, CNN, and Transformer backbones
vae/train.py                Training entry point
vae/evaluate.py             Test-set evaluation
vae/sample.py               Prior sampling
vae/visualize.py            Figures and latent interpolation
docs/                       Short conceptual notes
assets/                     Published figures and metrics
```

## Documentation

- [VAE overview](docs/vae_overview.md)
- [ELBO](docs/elbo.md)
- [Reparameterization](docs/reparameterization.md)
- [Flow prior](docs/flow_prior.md)

## Takeaway

The repository keeps the probabilistic route fixed while making architectural
and prior choices replaceable. The encoder learns an approximate posterior, the
ELBO balances reconstruction with latent regularization, and the decoder turns
prior samples into images.

```math
\mathrm{sample}
=
\mathrm{Decoder}_\theta(z),
\qquad
z\sim p_\psi(z).
```

## License

MIT.
