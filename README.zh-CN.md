# Modular VAE

<p align="right">
<a href="README.md">English</a> | 中文
</p>

这份仓库用一组小而清楚的 PyTorch 模块来梳理 VAE。编码器、解码器和先验都可以替换，便于单独观察每个选择会带来什么变化。

```text
图像 -> 编码器 -> 后验 q(z|x) -> 潜变量 z -> 解码器 -> 重建图像
                              |
                           先验 p(z)
```

所有模型都沿用同一套概率接口；实验之间只更换编码器、解码器、先验或 ELBO 配置。这样读代码时更容易看清各个模块的职责，做实验时也能单独比较某项改动的影响。

## 基本想法

普通自编码器把图像压缩成一个确定向量，VAE 则让编码器给出一个近似后验分布：

```math
q_\phi(z \mid x)
=
\mathcal{N}\left(
z;
\mu_\phi(x),
\mathrm{diag}(\sigma_\phi^2(x))
\right).
```

为了让这个随机采样仍然能够反向传播，VAE 把随机性单独写成 `epsilon`，这就是重参数化技巧：

```math
\epsilon \sim \mathcal{N}(0,I),
\qquad
z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon.
```

解码器再把采样得到的潜变量还原为图像：

```math
z \sim q_\phi(z\mid x),
\qquad
\hat{x}=\mathrm{Decoder}_\theta(z).
```

模型共用的概率接口在 [`vae/models/base_vae.py`](vae/models/base_vae.py)，对角高斯后验则在 [`vae/distributions.py`](vae/distributions.py)。

## 训练目标

训练时最小化的是负 ELBO，它由重建项和 KL 项两部分组成。重建项要求输出尽量接近输入，KL 项则防止每张图像占据彼此割裂的潜空间区域。

```math
\mathcal{L}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[-\log p_\theta(x\mid z)\right]
+
\beta\,
\mathrm{KL}\left(q_\phi(z\mid x)\Vert p_\psi(z)\right).
```

当先验是标准高斯时，对角高斯之间的 KL 散度可以直接算出：

```math
\mathrm{KL}\left(q_\phi(z\mid x)\Vert\mathcal{N}(0,I)\right)
=
\frac{1}{2}
\sum_j
\left(
\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2
\right).
```

如果先验本身需要学习，或者不再是标准高斯，就改用后验样本来估计 KL：

```math
\mathrm{KL}\left(q_\phi(z\mid x)\Vert p_\psi(z)\right)
\approx
\log q_\phi(z\mid x)-\log p_\psi(z),
\qquad
z\sim q_\phi(z\mid x).
```

目标函数实现在 [`vae/losses.py`](vae/losses.py)。

## 采样

生成时不再需要输入图像：从先验中采一个潜变量，直接交给解码器即可。

```math
z\sim p_\psi(z),
\qquad
x\sim p_\theta(x\mid z).
```

标准实验使用固定高斯先验：

```math
p(z)=\mathcal{N}(0,I).
```

流先验实验用 RealNVP 风格的仿射耦合层，把标准高斯基变量变换成更灵活的先验：

```math
u\sim\mathcal{N}(0,I),
\qquad
z=f_\psi(u).
```

对应的概率密度通过逆变换计算：

```math
u=f_\psi^{-1}(z),
\qquad
\log p_\psi(z)
=
\log p_0(u)
+
\log\left|\det\frac{\partial u}{\partial z}\right|.
```

先验接口在 [`vae/priors.py`](vae/priors.py)，可逆流的具体实现在 [`vae/flows.py`](vae/flows.py)。

## 实验设置

所有实验都使用 MNIST，也都遵循“后验—采样—解码”这条路径。每次只替换需要考察的组件，其他部分保持不变。

| 实验 | 骨干网络 | 先验 | 目标 | 用途 |
| --- | --- | --- | --- | --- |
| MLP VAE | MLP | 标准正态 | 解析 KL，`beta=1` | 全连接基线 |
| CNN VAE | CNN | 标准正态 | 解析 KL，`beta=1` | 图像结构归纳偏置 |
| Beta-VAE | MLP | 标准正态 | 解析 KL，`beta=4` | 加强潜空间约束 |
| Transformer VAE | Patch Transformer | 标准正态 | 解析 KL，`beta=1` | 使用基于图像块的注意力骨干 |
| 流先验 VAE | MLP | RealNVP 风格流 | Monte Carlo KL，`beta=1` | 学习非高斯先验 |

模块边界如下：

| 组件 | 实现 |
| --- | --- |
| 对角高斯后验 | [`vae/distributions.py`](vae/distributions.py) |
| 标准先验和流先验 | [`vae/priors.py`](vae/priors.py) |
| RealNVP 风格可逆流 | [`vae/flows.py`](vae/flows.py) |
| MLP、CNN 和 Transformer 编码器 | [`vae/models/encoders.py`](vae/models/encoders.py) |
| MLP、CNN 和 Transformer 解码器 | [`vae/models/decoders.py`](vae/models/decoders.py) |
| ELBO 和 beta-VAE 目标 | [`vae/losses.py`](vae/losses.py) |
| 配置驱动的构建器 | [`vae/builders.py`](vae/builders.py) |

## 和其他生成方法放在一起看

| 方法 | 学习对象 | 生成方式 |
| --- | --- | --- |
| VAE | 近似后验、先验和解码器 | 采样潜变量后解码 |
| Diffusion | 噪声、score、数据或速度 | 反向 SDE、ODE 或离散去噪链 |
| Flow Matching | 连续速度场 | 从噪声出发解 ODE |
| 这份实现 | 可替换的 VAE 骨干、先验和 ELBO 配置 | 潜变量采样和 MNIST 解码 |

VAE 从潜空间采样后，只需经过一次解码就能得到图像。Diffusion 和 Flow Matching 则要在像素空间或潜空间中运行一段逐步演化的过程。

## 结果

下表列出五组 MNIST 实验在测试集上的损失组成。

| 实验 | 骨干网络 | 先验 | KL 模式 | Loss | Recon | KL |
| --- | --- | --- | --- | ---: | ---: | ---: |
| MLP VAE | MLP | 标准正态 | 解析 | 100.05 | 79.98 | 20.07 |
| CNN VAE | CNN | 标准正态 | 解析 | 95.98 | 75.31 | 20.67 |
| Beta-VAE | MLP | 标准正态 | 解析 | 142.58 | 107.85 | 8.68 |
| Transformer VAE | Patch Transformer | 标准正态 | 解析 | 96.04 | 73.92 | 22.13 |
| 流先验 VAE | MLP | RealNVP 风格流 | Monte Carlo | 95.75 | 75.99 | 19.76 |

| 实验 | 重建 | 先验采样 |
| --- | --- | --- |
| MLP VAE | <img src="assets/figures/mlp_reconstructions.png" width="260" alt="MLP VAE 重建"> | <img src="assets/figures/mlp_samples_from_prior.png" width="260" alt="MLP VAE 先验采样"> |
| CNN VAE | <img src="assets/figures/cnn_reconstructions.png" width="260" alt="CNN VAE 重建"> | <img src="assets/figures/cnn_samples_from_prior.png" width="260" alt="CNN VAE 先验采样"> |
| Beta-VAE | <img src="assets/figures/beta_vae_reconstructions.png" width="260" alt="Beta-VAE 重建"> | <img src="assets/figures/beta_vae_samples_from_prior.png" width="260" alt="Beta-VAE 先验采样"> |
| Transformer VAE | <img src="assets/figures/transformer_reconstructions.png" width="260" alt="Transformer VAE 重建"> | <img src="assets/figures/transformer_samples_from_prior.png" width="260" alt="Transformer VAE 先验采样"> |
| 流先验 VAE | <img src="assets/figures/flow_prior_reconstructions.png" width="260" alt="流先验 VAE 重建"> | <img src="assets/figures/flow_prior_samples_from_prior.png" width="260" alt="流先验 VAE 先验采样"> |

完整的 YAML 指标文件保存在 [`assets/results/`](assets/results/)。

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
vae/priors.py               标准正态和流先验
vae/flows.py                RealNVP 风格仿射耦合流
vae/losses.py               ELBO 和 beta-VAE 目标
vae/builders.py             配置驱动的组件构建
vae/models/                 MLP、CNN 和 Transformer 骨干
vae/train.py                训练入口
vae/evaluate.py             测试集评估
vae/sample.py               先验采样
vae/visualize.py            图像与潜空间插值
docs/                       概念说明
assets/                     已发布图像和指标
```

## 文档

- [VAE 概览](docs/vae_overview.md)
- [ELBO](docs/elbo.md)
- [重参数化](docs/reparameterization.md)
- [流先验](docs/flow_prior.md)

## 小结

这份实现把 VAE 中稳定不变的概率流程，与可以自由替换的模型组件分开。编码器给出近似后验，ELBO 在重建质量和潜空间约束之间做权衡，解码器则负责把先验样本变成图像。

```math
\mathrm{sample}
=
\mathrm{Decoder}_\theta(z),
\qquad
z\sim p_\psi(z).
```

## License

MIT.
