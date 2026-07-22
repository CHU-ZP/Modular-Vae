# Modular VAE

<p align="right">
<a href="README.md">English</a> | 中文
</p>

这是一个紧凑的 VAE 实践项目，把概率 latent 表示学习映射到小而清晰的 PyTorch 模块中，并支持替换骨干网络和先验。

```text
图像 -> 编码器 -> 后验 q(z|x) -> latent z -> 解码器 -> 重建图像
                              |
                           先验 p(z)
```

仓库面向学习和实验：概率接口保持不变，编码器、解码器、先验和 ELBO 配置则作为显式的实验选择。

## 基本想法

变分自编码器不会把图像映射到一个确定向量。编码器参数化的是一个近似后验分布：

```math
q_\phi(z \mid x)
=
\mathcal{N}\left(
z;
\mu_\phi(x),
\mathrm{diag}(\sigma_\phi^2(x))
\right).
```

采样使用重参数化技巧，使梯度可以穿过后验参数：

```math
\epsilon \sim \mathcal{N}(0,I),
\qquad
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon.
```

解码器再把采样得到的 latent 映射回图像：

```math
z \sim q_\phi(z\mid x),
\qquad
\hat{x}=\mathrm{Decoder}_\theta(z).
```

共享概率封装实现在 [`vae/models/base_vae.py`](vae/models/base_vae.py)，对角高斯后验实现在 [`vae/distributions.py`](vae/distributions.py)。

## 训练目标

训练最小化负证据下界：重建项负责恢复输入，KL 正则项则把近似后验约束到先验附近。

```math
\mathcal{L}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[-\log p_\theta(x\mid z)\right]
+
\beta\,
\mathrm{KL}\left(q_\phi(z\mid x)\Vert p_\psi(z)\right).
```

对于标准高斯先验，对角高斯 KL 可以解析计算：

```math
\mathrm{KL}\left(q_\phi(z\mid x)\Vert\mathcal{N}(0,I)\right)
=
\frac{1}{2}
\sum_j
\left(
\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2
\right).
```

对于学习得到的或非标准先验，KL 使用一个后验样本估计：

```math
\mathrm{KL}\left(q_\phi(z\mid x)\Vert p_\psi(z)\right)
\approx
\log q_\phi(z\mid x)-\log p_\psi(z),
\qquad
z\sim q_\phi(z\mid x).
```

目标函数实现在 [`vae/losses.py`](vae/losses.py)。

## 采样

生成时不需要输入图像。只需从先验中采样 latent，再直接交给解码器：

```math
z\sim p_\psi(z),
\qquad
x\sim p_\theta(x\mid z).
```

标准实验使用固定高斯先验：

```math
p(z)=\mathcal{N}(0,I).
```

flow-prior 实验使用 RealNVP 风格的仿射耦合流变换标准高斯基变量：

```math
u\sim\mathcal{N}(0,I),
\qquad
z=f_\psi(u).
```

它的密度通过逆变换计算：

```math
u=f_\psi^{-1}(z),
\qquad
\log p_\psi(z)
=
\log p_0(u)
+
\log\left|\det\frac{\partial u}{\partial z}\right|.
```

先验实现在 [`vae/priors.py`](vae/priors.py)，flow 实现在 [`vae/flows.py`](vae/flows.py)。

## 这个项目实现了什么

所有实验都使用 MNIST，并保持相同的“后验—采样—解码”路径。每组实验只替换对照所需的组件。

| 实验 | 骨干网络 | 先验 | 目标 | 用途 |
| --- | --- | --- | --- | --- |
| MLP VAE | MLP | 标准正态 | 解析 KL，`beta=1` | 全连接基线 |
| CNN VAE | CNN | 标准正态 | 解析 KL，`beta=1` | 图像结构归纳偏置 |
| Beta-VAE | MLP | 标准正态 | 解析 KL，`beta=4` | 更强的 latent 正则化 |
| Transformer VAE | Patch Transformer | 标准正态 | 解析 KL，`beta=1` | patch-based attention 骨干 |
| Flow-prior VAE | MLP | RealNVP 风格 flow | Monte Carlo KL，`beta=1` | 学习非高斯先验 |

模块边界如下：

| 组件 | 实现 |
| --- | --- |
| 对角高斯后验 | [`vae/distributions.py`](vae/distributions.py) |
| 标准先验和 flow prior | [`vae/priors.py`](vae/priors.py) |
| RealNVP 风格 flow | [`vae/flows.py`](vae/flows.py) |
| MLP、CNN 和 Transformer 编码器 | [`vae/models/encoders.py`](vae/models/encoders.py) |
| MLP、CNN 和 Transformer 解码器 | [`vae/models/decoders.py`](vae/models/decoders.py) |
| ELBO 和 beta-VAE 目标 | [`vae/losses.py`](vae/losses.py) |
| 配置驱动的构建器 | [`vae/builders.py`](vae/builders.py) |

## 和邻近方法的关系

| 方法 | 学习对象 | 生成方式 |
| --- | --- | --- |
| VAE | 近似后验、先验和解码器 | 采样 latent 后解码 |
| Diffusion | noise、score、data 或 velocity prediction | reverse SDE、ODE 或去噪链 |
| Flow Matching | 连续速度场 | 从噪声出发解 ODE |
| 本项目 | 可替换的 VAE 骨干、先验和 ELBO 配置 | latent 采样和 MNIST 解码 |

VAE 在采样紧凑 latent 之后只需一次解码即可生成。Diffusion 和 Flow Matching 则会在 pixel 或 latent space 中学习迭代动力学。

## 结果

下表给出五组已完成 MNIST 实验的测试集 loss 分量。

| 实验 | 骨干网络 | 先验 | KL 模式 | Loss | Recon | KL |
| --- | --- | --- | --- | ---: | ---: | ---: |
| MLP VAE | MLP | 标准正态 | 解析 | 100.05 | 79.98 | 20.07 |
| CNN VAE | CNN | 标准正态 | 解析 | 95.98 | 75.31 | 20.67 |
| Beta-VAE | MLP | 标准正态 | 解析 | 142.58 | 107.85 | 8.68 |
| Transformer VAE | Patch Transformer | 标准正态 | 解析 | 96.04 | 73.92 | 22.13 |
| Flow-prior VAE | MLP | RealNVP 风格 flow | Monte Carlo | 95.75 | 75.99 | 19.76 |

| 实验 | 重建 | 先验采样 |
| --- | --- | --- |
| MLP VAE | <img src="assets/figures/mlp_reconstructions.png" width="260" alt="MLP VAE 重建"> | <img src="assets/figures/mlp_samples_from_prior.png" width="260" alt="MLP VAE 先验采样"> |
| CNN VAE | <img src="assets/figures/cnn_reconstructions.png" width="260" alt="CNN VAE 重建"> | <img src="assets/figures/cnn_samples_from_prior.png" width="260" alt="CNN VAE 先验采样"> |
| Beta-VAE | <img src="assets/figures/beta_vae_reconstructions.png" width="260" alt="Beta-VAE 重建"> | <img src="assets/figures/beta_vae_samples_from_prior.png" width="260" alt="Beta-VAE 先验采样"> |
| Transformer VAE | <img src="assets/figures/transformer_reconstructions.png" width="260" alt="Transformer VAE 重建"> | <img src="assets/figures/transformer_samples_from_prior.png" width="260" alt="Transformer VAE 先验采样"> |
| Flow-prior VAE | <img src="assets/figures/flow_prior_reconstructions.png" width="260" alt="Flow-prior VAE 重建"> | <img src="assets/figures/flow_prior_samples_from_prior.png" width="260" alt="Flow-prior VAE 先验采样"> |

完整指标 YAML 文件位于 [`assets/results/`](assets/results/)。

## 快速使用

先安装适合当前设备的 PyTorch，再安装其余依赖：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

运行全部实验：

```bash
bash scripts/run_all_experiments.sh
```

运行单个实验：

```bash
python -m vae.train --config configs/mnist_mlp_standard.yaml
```

生成评估指标和图像：

```bash
python -m vae.evaluate --checkpoint outputs/mnist_mlp_standard/checkpoint.pt
python -m vae.visualize --checkpoint outputs/mnist_mlp_standard/checkpoint.pt
```

## 项目结构

```text
configs/                    YAML 实验配置
vae/distributions.py        对角高斯后验
vae/priors.py               标准正态和 flow prior
vae/flows.py                RealNVP 风格仿射耦合流
vae/losses.py               ELBO 和 beta-VAE 目标
vae/builders.py             配置驱动的组件构建
vae/models/                 MLP、CNN 和 Transformer 骨干
vae/train.py                训练入口
vae/evaluate.py             测试集评估
vae/sample.py               先验采样
vae/visualize.py            图像和 latent 插值
docs/                       概念说明
assets/                     已发布图像和指标
```

## 文档

- [VAE 概览](docs/vae_overview.md)
- [ELBO](docs/elbo.md)
- [重参数化](docs/reparameterization.md)
- [Flow prior](docs/flow_prior.md)

## 小结

这个仓库保持概率路径不变，同时使架构和先验可替换。编码器学习近似后验，ELBO 在重建与 latent 正则化之间取得平衡，解码器再把先验样本转换成图像。

```math
\mathrm{sample}
=
\mathrm{Decoder}_\theta(z),
\qquad
z\sim p_\psi(z).
```

## License

MIT。
