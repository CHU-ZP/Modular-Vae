# Modular VAE Demo

This project is a small PyTorch research demo for Variational Autoencoders. The fixed part is the probabilistic interface:

```text
x -> Encoder -> mu, logvar -> q_phi(z|x) -> z -> Decoder -> x_hat
                                |
                              p_psi(z)
```

The replaceable parts are the encoder backbone, decoder backbone, prior, and ELBO configuration.

## What This Demonstrates

A VAE does not encode an image `x` into one deterministic vector. The encoder outputs parameters of a posterior distribution:

```text
q_phi(z|x) = N(mu(x), diag(sigma^2(x)))
```

Sampling uses the reparameterization trick:

```text
z = mu + sigma * epsilon,   epsilon ~ N(0, I)
```

That keeps sampling differentiable with respect to `mu` and `sigma`. The decoder then models `p_theta(x|z)` by reconstructing the image from `z`.

The KL term aligns the posterior `q_phi(z|x)` with a prior `p_psi(z)`. With a standard Gaussian prior, the KL has a closed form. With a flow prior, the loss uses Monte Carlo KL:

```text
KL(q_phi(z|x) || p_psi(z)) ~= log q_phi(z|x) - log p_psi(z)
```

The Transformer VAE here is not a different probabilistic model. It is only a different encoder/decoder backbone behind the same posterior, sampling, prior, and ELBO interface.

## Project Structure

```text
configs/                  YAML experiment configs
vae/distributions.py       DiagonalGaussian posterior helper
vae/priors.py              StandardNormalPrior and FlowPrior
vae/flows.py               RealNVP-style affine coupling flow
vae/losses.py              ELBO / beta-VAE loss
vae/builders.py            Config-driven component builders
vae/models/                MLP, CNN, and Transformer VAE backbones
vae/data/mnist.py          MNIST dataloaders
vae/train.py               Training entry point
vae/sample.py              Prior sampling entry point
vae/evaluate.py            Test-set evaluation
vae/visualize.py           Reconstruction, sampling, interpolation, curves
docs/                      Short conceptual notes
```

## Install

```bash
pip install -r requirements.txt
```

## Experiments

Train the standard MLP VAE:

```bash
python -m vae.train --config configs/mnist_mlp_standard.yaml
```

Train the CNN VAE:

```bash
python -m vae.train --config configs/mnist_cnn_standard.yaml
```

Train beta-VAE:

```bash
python -m vae.train --config configs/mnist_beta_vae.yaml
```

Train the patch Transformer VAE:

```bash
python -m vae.train --config configs/mnist_transformer_standard.yaml
```

Train the flow-prior VAE:

```bash
python -m vae.train --config configs/mnist_flow_prior.yaml
```

Outputs are written to `outputs/<experiment_name>/`:

```text
checkpoint.pt
training_curves.png
reconstructions.png
```

## Sampling

```bash
python -m vae.sample --checkpoint outputs/mnist_mlp_standard/checkpoint.pt
```

For a standard VAE, this samples `z ~ N(0, I)`. For a flow-prior VAE, it samples `u ~ N(0, I)` and maps `z = f_psi(u)` before decoding.

## Visualization

```bash
python -m vae.visualize --checkpoint outputs/mnist_mlp_standard/checkpoint.pt
```

This writes:

```text
reconstructions.png
samples_from_prior.png
latent_interpolation.png
training_curves.png
```

If `latent_dim = 2`, it also writes `latent_space_2d.png`, plotting posterior means `mu(x)`.

## Configuration

The core knobs are:

```yaml
model:
  encoder: mlp        # mlp | cnn | transformer
  decoder: mlp        # mlp | cnn | transformer
  latent_dim: 16

prior:
  type: standard_normal  # standard_normal | flow

loss:
  beta: 1.0
  kl_mode: analytic      # analytic | monte_carlo
```

Use `kl_mode: analytic` with `StandardNormalPrior`. Use `kl_mode: monte_carlo` when the prior is learned or nonstandard, such as `FlowPrior`.
