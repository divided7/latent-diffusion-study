# 文生图大模型——Stable Diffusion

**时间:** 2024-07-31   **版本:** 0.1.0

---

Stable Diffusion 是一种文本到图像的潜在扩散模型，由CompVis、Stability AI和LAION的研究人员和工程师创建。它使用来自LAION-5B数据库子集的 512x512 图像进行训练。LAION-5B是目前最大的、可免费访问的多模态数据集。

事实上，文生图的算法也有利用GAN或者其它方式，但最主流的还是Stable diffusion这种以扩散模型实现。本文主要介绍Stable diffusion模型，对于其它模式的文生图算法不做赘述。

关于生成式AI的前期文档：[生成式模型概述](http://172.16.200.150:7100/generative_model_summary)

## **Stable Diffusion v1.0 （原名Latent-diffusion）**

- **Github**：[Stable Diffusion v1.0 GitHub](https://github.com/CompVis/latent-diffusion)
- **论文链接**：[Stable Diffusion: A High-Resolution Image Synthesis Model](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) （CVPR）
- **论文链接：**[Stable Diffusion: A High-Resolution Image Synthesis Model](https://arxiv.org/pdf/2112.10752) （Arxiv, 完整版论文）

**Stable Diffusion v1.x**

后续的1.x版本都是在1.0的基础上进行改进： 

1. **sd-v1-1.ckpt**
   - **训练步骤**：237k步骤，分辨率256x256，使用了LAION-2B-en数据集。194k步骤，分辨率512x512，使用了LAION高分辨率数据集（来自LAION-5B中分辨率大于等于1024x1024的170M样本）。
2. **sd-v1-2.ckpt**
   - **训练步骤**：从sd-v1-1.ckpt继续。515k步骤，分辨率512x512，使用了LAION-Aesthetics v2 5+数据集（LAION-2B-en的子集，图像的估计美学评分大于5.0，原始尺寸大于等于512x512，并且估计水印概率小于0.5。水印估计来自LAION-5B的元数据，美学评分使用LAION-Aesthetics Predictor V2进行估计）。
3. **sd-v1-3.ckpt**
   - **训练步骤**：从sd-v1-2.ckpt继续。195k步骤，分辨率512x512，使用了“LAION-Aesthetics v2 5+”数据集，并且在生成过程中降低了10%的文本条件以改进无分类器指导采样（classifier-free guidance sampling）。
4. **sd-v1-4.ckpt**
   - **训练步骤**：从sd-v1-2.ckpt继续。225k步骤，分辨率512x512，使用了“LAION-Aesthetics v2 5+”数据集，并且在生成过程中降低了10%的文本条件以改进无分类器指导采样（classifier-free guidance sampling）。
5. **sd-v1-5.ckpt**
   - **训练步骤**：从sd-v1-2.ckpt继续。595k步骤，分辨率512x512，使用了“LAION-Aesthetics v2 5+”数据集，并且在生成过程中降低了10%的文本条件以改进无分类器指导采样（classifier-free guidance sampling）。
6. **sd-v1-5-inpainting.ckpt**
   - **训练步骤**：从sd-v1-5.ckpt继续。440k步骤的修复训练，分辨率512x512，使用了“LAION-Aesthetics v2 5+”数据集，并且在生成过程中降低了10%的文本条件以改进无分类器指导采样（classifier-free guidance sampling）。对于修复任务，UNet有5个额外的输入通道（4个用于编码的遮罩图像，1个用于遮罩本身），其权重在恢复非修复检查点后被零初始化。在训练过程中，生成合成遮罩，并在25%的图像上遮挡所有内容。

## **Stable diffusion v2.x**

- **官网简介**：[Stable Diffusion v2.0](https://stability.ai/news/stable-diffusion-v2-release)
- **论文链接：**[SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) （Arxiv, 完整版论文）

## **Stable diffusion v3.x**

由stability.ai提供，在[Stable-diffusion-3](https://stability.ai/news/stable-diffusion-3)中提到 "We will publish a detailed technical report soon."，暂未公布技术细节。

# High-Resolution Image Synthesis with Latent Diffusion Models（Diffusion Model v1.0）

## Intro：

作者提到，DMs属于一种Likelihood-based Model(基于似然函数的模型)，这类模型通过最大化似然函数来进行参数估计。DMs 特别关注在数据的所有可能模式（或“模式覆盖”）上进行建模，这可能导致模型过度拟合数据中的细节，即使这些细节对最终任务并不重要。此外消耗的计算资源也很大，另外作者也提到了碳排问题。

作者首先对当时有的扩散模型进行分析。具体来说，对已经训练好的扩散模型在像素空间中的性能进行分析。像素空间指的是原始图像的空间，其中每个像素值都被直接使用来训练模型。作者提出可以**分为两个阶段：**perceptual compression 和 semantic compression 两个部分，即**感知压缩阶段**（这里应该是作者自己起的名字，并不是压缩感知领域的专有名词）和**语义压缩阶段**。其中**感知压缩阶段**是在训练初期去除不必要的高频特征（图像特别细节的一些内容），专注于主要视觉特征；**语义压缩阶段**是在模型的后期阶段学习数据的语义结构。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240731154924161.png" alt="image-20240731154924161" style="zoom:50%;" />

作者分别在语义维度和压缩维度进行展示，由上图的perceptual compression部分可知，即使对图像做了很大程度的感知压缩，图像的整体仍然为一个戴着墨镜的男人，只是细节发生改变；而在语义压缩阶段只是对语义维度稍微增加偏置就对结果产生巨大影响，由一个带墨镜的男人变成了一个女人。

所以作者给出的方案是：准备好一个感知压缩模型（或者理解为特征压缩模型、图像压缩模型），这里选用的是一个AE（自编码器），以提供一个低维的空间表示。接着将AE的Latent部分作为感知压缩的输出，送入语义压缩阶段。

**Contribution:**

(i) 提出的方法在处理高维数据时表现得更为高效，因此可以在提供比之前的工作更为真实和详细的重建的压缩水平上工作，并且可以高效地应用于百万像素图像的高分辨率合成。

(ii) 在多个任务（如无条件图像合成、图像修复、随机超分辨率）和数据集上实现了具有竞争力的性能，同时显著降低了计算成本。与基于像素的扩散方法相比还显著降低了推理成本。

(iii) 与之前的工作——同时学习编码器/解码器架构和基于评分的先验的做法不同，作者提出的的方法不需要精细地调整重建和生成能力的权重。这确保了极其真实的重建，并且对潜在空间的正则化需求非常小。

(iv) 对于密集条件的任务，如超分辨率、图像修复和语义合成，可以以卷积方式应用，并生成大规模、一致性的图像（约 1024×1024像素）。

(v) 设计了一种基于交叉注意力的通用条件机制，实现了多模态训练，使用它来训练类别条件模型、文本到图像模型以及布局到图像模型。

(vi)发布了预训练的LDM模型和AE模型，这些模型可能在训练扩散模型之外的各种任务中也可重复使用。

## Method:

尽管扩散模型通过忽略与感知无关的细节（即通过欠采样相应的损失项）来减少计算量，但它们在像素空间中的函数评估仍然非常耗费计算时间和能源资源。

为了解决这一问题，通过明确区分压缩学习阶段和生成学习阶段来绕过这一缺陷。为此利用一个自编码模型，该模型学习一个在感知上等效于图像空间但计算复杂度显著降低的空间。

总的来说：

(i) 通过远离高维的图像空间，获得了计算效率更高的扩散模型，因为采样是在低维空间中进行的。

(ii) 利用了扩散模型从其UNet架构继承的归纳偏置，使其在处理具有空间结构的数据时特别有效，因此缓解了现有方法所需的激进且会降低质量的压缩水平的需求。

(iii)获得了通用的压缩模型，其潜在空间可以用于训练多种生成模型，并且还可以用于其他下游应用，如单图像CLIP引导的合成。

### **Perceptual Image Compression**

感知压缩模型是基于之前的工作实现的 ，由一个自编码器AE组成，AE通过组合perceptual loss[^感知损失]:和patch-based[^patch-based]的对抗目标进行训练。作者认为这种方法确保了重建图像被限制在图像流形上，通过强制局部真实性来避免仅依赖像素空间损失（如L2或L1目标）所引入的模糊。

[^感知损失]: 参考论文:[The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf ), perceptual loss是一种用于图像合成、风格迁移等任务中的损失函数，其目标是衡量生成图像与真实图像在感知上的相似性，而不仅仅是像素级的差异。传统的图像质量评估指标如 PSNR和 SSIM虽然简单易用，但无法全面捕捉人类视觉系统的复杂感知特性。这些传统指标通常是基于像素的浅层函数，未能反映图像的高级语义和结构信息。感知损失的核心思想是利用深度学习模型的特征，特别是深层卷积神经网络（如 VGG 网络）提取的特征图，这些特征图能够更好地表示图像的高级语义和结构特征。这种损失函数通过比较生成图像和目标图像在深度网络中提取的特征差异来衡量图像的感知相似性，而不是仅仅比较像素级的差异。
[^patch-based]: [Image-to-image translation with conditional adversarial networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf) ，在 patch-based 方法中，整个图像被划分成若干个小的局部Block或者patch。每个patch都被单独处理，这种局部处理可以捕捉到图像中的细节特征，尤其是在图像修复、纹理合成等任务中非常有用。换言之，细粒度从像素变成了patch。

具体来说：

- 给定图像 $ x \in \mathbb{R}^{H \times W \times 3} $

- 编码器 $E$ 将图像 $x$ 编码为一个潜在表示  $z$ ：  $z = E(x)$ ,  $z \in \mathbb{R}^{h \times w \times c}$.  $h,w,c$ 为Latent的高宽和通道 

- 解码器 $D$ 从潜在表示 $z$ 中重建图像，得到重建图像 $\tilde{x}$ ：  $\tilde{x} = D(z) = D(E(x))$

- 编码器在编码过程中对图像进行下采样，下采样因子为 $f$ ：   $ f = \frac{H}{h} = \frac{W}{w} $, $一般有f=2^m,m \in \mathbb{N}$  。

同时为了避免Latent方差过大导致图像质量低，使用了两种正则化方法：

**KL-正则化（KL-reg.）**：

- KL-正则化[^kl-reg]是通过向潜在空间施加一个轻微的 Kullback-Leibler（KL）罚项，使得潜在空间的分布接近标准正态分布。这种方法类似于变分自编码器（VAE）中的正则化策略。通过施加 KL 罚项，可以使潜在空间的分布更加规范化，从而降低潜在空间的方差，避免潜在空间的过度复杂性。公式如下： $ \text{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$

  [^kl-reg]: 其实意思是总损失由两个损失构成，一个是使得编码器的结果Latent输出的向量服从正态分布，一个是使得解码器输出的图像和输入图像相同，伪代码如下

  ```python
  encoder, decoder = Encoder(), Decoder() # 实例化模型
  for epoch in range(num_epochs):
      for data in dataloader:
          x = data  # 输入图像
          z_mean, z_log_var = encoder(x)  # 编码器输出潜在表示的均值和对数方差
          z = sample_latent(z_mean, z_log_var)  # 从潜在分布中采样潜在变量
          x_reconstructed = decoder(z)  # 解码器生成重建图像
          reconstruction_loss = calculate_reconstruction_loss(x, x_reconstructed) # 计算重建损失(即期望解码器输出和输入相同)
          kl_divergence_loss = calculate_kl_divergence(z_mean, z_log_var) # 计算 KL 散度损失
          total_loss = reconstruction_loss + lambda_kl * kl_divergence_loss # 计算总损失
          ...
  # 关于encoder模型这里顺带提一下，encoder并不是输出特征图然后通过公式去计算均值和方差，而是使用线性层映射到mean和var，模型伪代码如下：
  class Encoder(nn.Module):
      def __init__(self, in_channels, latent_dim):
          super(Encoder, self).__init__()
          self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
          self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
          self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
  
          self.fc1 = nn.Linear(128 * 16 * 16, 256)  # 隐藏层
          self.fc_mean = nn.Linear(256, latent_dim)  # 潜在空间均值
          self.fc_log_var = nn.Linear(256, latent_dim)  # 潜在空间对数方差
  
      def forward(self, x):
          x = F.relu(self.conv1(x))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = x.view(-1, 128 * 16 * 16) # x = x.flatten()
          x = F.relu(self.fc1(x))
          z_mean = self.fc_mean(x)  # 潜在空间均值
          z_log_var = self.fc_log_var(x)  # 潜在空间对数方差
          return z_mean, z_log_var
  # 使用线性层映射而不直接计算的原因：
  # 1. 学习到的特征与潜在变量的关系
  # 线性层允许模型通过学习到的特征与潜在空间的均值和方差之间建立一个复杂的映射。直接从特征图计算均值和方差可能无法捕捉到这种复杂的关系。线性层通过参数化的方式可以在特征空间中找到更合适的表示。
  # 2. 减少计算复杂度
  # 计算均值和方差通常会涉及到全局平均池化或其他操作，这可能会增加计算复杂度和内存使用。线性层提供了一个简洁的方式来实现这一功能，尤其是在处理高维特征图时。
  # 3. 灵活性与可训练性
  # 通过线性层，网络可以在训练过程中学习如何将特征图的复杂模式映射到潜在空间的均值和方差。这种方法提供了更大的灵活性，因为模型可以调整这些线性层的权重来适应训练数据的特性。
  # 4. 确保潜在空间的高效性
  # 线性层提供了一种有效的方式来映射到潜在空间。这种方法可以确保潜在空间的均值和方差是基于特征图学习到的，而不是简单地从特征图计算得到，从而使得潜在空间更具表达能力。
  # 5. 模型的可解释性和稳定性
  # 在某些情况下，直接计算均值和方差可能会导致数值不稳定性或训练难度加大。使用线性层可以在一定程度上缓解这些问题，因为它通过学习得到的权重来生成均值和方差，从而提供了更多的稳定性和解释性。
  ```

**矢量量化正则化（VQ-reg.）**：

- 矢量量化正则化使用了一个矢量量化层（vector quantization layer），该层在解码器中实现。这种模型可以被解释为 VQGAN，但其中的量化层被吸收到了解码器中。矢量量化层将潜在空间的表示量化为离散的代码簇，有助于保持潜在表示的结构化，并减少冗余信息。这种方法可以更有效地捕捉和复原图像的细节。（详细可以参考[生成式模型概述](http://172.16.200.150:7100/generative_model_summary)中VQ-VAE的内容, 更接近VQGAN的思想）

```python
class VQDecoder(nn.Module):
    def __init__(self, in_channels, latent_dim, num_embeddings):
        ...
        def forward(self, x):
        ...
        z_flatten = z.view(-1, self.latent_dim)  # 展平特征
        distances = torch.cdist(z_flatten, self.embedding.weight)  # 计算距离
        indices = torch.argmin(distances, dim=1)  # 找到最近的码本向量
        z_quantized = self.embedding(indices)  # 使用量化向量
        # 计算量化损失
        loss = self.vq_criterion(z_flatten, z_quantized)
        return z_quantized, loss
```

由于模型的设计考虑到了Latent空间 $ z=E(x) $ 的二维结构，模型能够使用相对温和的压缩率来实现非常好的重建效果。这与以往工作不同，以往的工作可能依赖于对潜在空间 $z$ 的一维排序，以Auto Regression方式建模其分布，这忽略了潜在空间的固有结构。通过保持潜在空间的二维结构，模型能够更好地保留图像 $ x $ 的细节，表现出更好的重建效果。这种方法在重建图像时相较于传统的方式能够减少信息损失，从而提高图像质量。

### Latent Diffusion Models

**Diffusion Model**

作者提到扩散模型是一种概率模型，其设计目的是通过逐步去噪一个正态分布的变量来学习数据分布 $p(x)$，这对应于学习一个固定长度 $T$ 的马尔可夫链的逆过程。

在图像合成中，最成功的模型依赖于reweighted variant of the variational lower bound (变分下界的重加权变体)，这与denoising score-matching (去噪分数匹配)相呼应。这些模型可以被解释为一系列等权重的去噪自编码器 $ \epsilon_{\theta}(x_t, t)$ ，其中  $t$  从 {1, ..., $T$} 中均匀采样，每个去噪自编码器被训练以预测其输入 \( $x_t$ \) 的去噪变体，$x_t$ 是输入图像 $x$ 的噪声版本。

对应的目标函数可以简化为：$ L_{\text{DM}} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_{\theta}(x_t, t) \|^2_2 \right]$ ，   $t$  从 {1, ..., $T$} 中均匀采样

**Generative Modeling of Latent Representations**

生成式模型的隐藏空间表示：**训练好的感知压缩模型**将输入图像转换为一个有效的低维潜在空间，在这个空间中高频、不可感知的细节被抽象掉了，**能够更加关注重要的语义信息**，而不是处理大量的不可感知细节，同时加速计算。与高维像素空间相比，这个空间更适合likelihood-based的生成模型，因为其更加专注于数据的重要语义位，并且在较低维度计算效率更高的空间中进行训练。

不同于先前的工作是依赖自回归的方法和基于注意力的transformer模型高度压缩离散潜在空间；在这篇文章中作者提出的方法是利用图像特有的inductive biases（归纳偏置），包括：**UNet结构**和**关注感知上最相关的部分**（通过重加权的下界来专注于最重要的感知信息）。

因此，重加权的目标函数表示为：

$L_{\text{LDM}} := \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_{\theta}(z_t, t) \|^2_2 \right]$

其中 $\mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t}$ 表示对数据 $x$、标准正态分布 $\epsilon$ 和时间步 $t$ 的期望，$\epsilon$ 是噪声， $\epsilon_{\theta}$ 是模型的去噪预测，$z_t$ 是从潜在空间中得到的表示。

模型的backbone: $ \epsilon_\theta(\cdot, t)$ 是通过一个时间条件的 UNet 实现的，由于前向过程是固定的，$z_t $可以在训练期间通过编码器 $E$ 高效地获得，而从 $p(z)$ 中采样的样本可以通过解码器 $D$ 一次性转换到图像空间。

### **Conditioning Mechanisms**

**条件机制**

与其他类型的生成模型类似，扩散模型原则上能够建模形式为 $p(z|y)$ 的条件分布。这可以通过条件去噪自编码器 $\epsilon_\theta(z_t, t, y)$ 来实现，从而使我们能够通过输入 $y$（例如文本、语义图或其他图像到图像的翻译任务）来控制合成过程。然而在图像合成的背景下将 DMs 的生成能力与超出类别标签或模糊变体的输入图像的其他条件类型结合起来，仍然是一个未被充分探索的研究领域。

作者团队通过在其基础 UNet 上增加跨注意力机制，将 DMs 转变为更灵活的条件图像生成器。跨注意力机制在学习基于各种输入模态的注意力模型时非常有效 [35, 36]。为了处理来自各种模态（如语言提示）的 $y$，我们引入了一个领域特定的编码器 $ \tau_\theta $ ，该编码器将 $y$ 投影到一个中间表示  $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau} $ ，然后通过一个跨注意力层将其映射到 UNet 的中间层，跨注意力层的实现为：$ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d}} \right) \cdot V $

其中：

- $Q = W^{(i)}_Q \cdot \phi_i(z_t)$
- $K = W^{(i)}_K \cdot \tau_\theta(y)$
- $V = W^{(i)}_V \cdot \tau_\theta(y)$

这里，\( $\phi_i(z_t) \in \mathbb{R}^{N \times d_i}$ \) 表示 UNet 实现 \( $\epsilon_\theta$ \) 的（展平）中间表示， $W^{(i)}_V \in \mathbb{R}^{d \times d_i} $，$ W^{(i)}_Q \in \mathbb{R}^{d \times d_\tau} $和  $W^{(i)}_K \in \mathbb{R}^{d \times d_\tau}$ 是可学习的投影矩阵，如下图所示。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240801164657150.png" alt="image-20240801164657150" style="zoom:50%;" />

基于image-conditioning pairs(图像-条件对)，通过以下方式学习条件 LDM：

$L_{LDM} := \mathbb{E}_{x, y, \epsilon \sim \mathcal{N}(0,1), t} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y)) \right\|_2^2 \right]$

其中 $\tau_\theta$ 和 $\epsilon_\theta$ 通过上面的损失$L_{LDM}$共同优化。这种条件机制是灵活的，因为 $\tau_\theta$ 可以通过领域特定的专家进行参数化，例如当 $y$ 是文本提示时，可以使用未mask的transformer 。



