# DFDGCN-Note

**一、论文来源**：2024_ICASSP_Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting

**二、研究动机**：

1. 时移 (Time-Shift)：时移会导致节点对的空间邻近性失效，这是因为欧式距离和余弦相似度等方法在对齐维度计算时，受时移影响的相互关联的交通数据的变化不会发生在同一时间戳，这增加了利用时序接近度来识别道路之间相关性的困难。例如下图中，位于两地的传感器具有序列相似性，但从 Sensor 155 传递或扩散的交通量需要经过半个小时或一刻钟才能达到 Sensor 157，这使得两点在同一时间戳上失配。

   <img src=".\images\QQ20240826-112322.png" style="zoom:80%;" />

2. 数据噪声 (Data Noise)：交通数据中的噪声是数据驱动策略中不可避免的问题，常常由交通事故或道路施工等紧急情况引发。因此带有噪声的交通数据并不总是可靠的。

**三、方法提出**：本文提出了一种动态频域图卷积网络 (DFDGCN) 用于捕捉复杂的空间依赖性。针对时移现象，依靠傅里叶变换的时移性质，交通数据频率分量仍处于相同的相位维度。针对数据噪声，学习传感器之间的空间关系的同时，引入额外的信息来减少噪声的影响，主要使用了传感器身份嵌入(identity embedding) 和时间嵌入 (time embedding)。

**四、理论介绍**：

论文采用的时序建模框架是基于 Graph WaveNet 的，在图结构学习上加入了自己的创新。具体来说，DFDGCN 的核心是根据当前观测窗口观测到的历史流量数据来更新动态邻接矩阵 $A_D$，如下图所示：

<img src=".\images\QQ20240826-113939.png" style="zoom:80%;" />

交通数据在频域受到时移的影响比时域要小。因此，先通过傅里叶变换将每个观测窗口的交通数据 $X_t$ 转移到频域，即 $F_t=FFT(X_t)$。

为了处理交通数据中的噪声，引入了节点身份嵌入和时间嵌入来增加有关交通网络的附加信息，以便可以轻松提取每个传感器有效的交通模式并探索它们之间的空间依赖关系。

$DE_t=W_{F,t} \cdot F_t || E_t || W_{T,t} \cdot (T_t^W||T_t^D)$

其中，$E_t$ 表示每个传感器可学习的身份嵌入，$T_t^W$ 和 $T_t^D$ 分别指示序列是星期几和一天中的什么时间戳，$W_{F,t}$ 和 $W_{T,t}$ 是可学习的参数矩阵。此外，文中还应用了 $1 \times 1$ 卷积核做进一步嵌入，以学习 $DE_t$ 维度之间的连接。

最后，为了学习 $DE_t$ 的方向性，将其与 $DE_t$ 的转置进行矩阵相乘，经过激活函数和 softmax 之后得到最终的邻接矩阵。

$A_D^t = Softmax(ReLU(DE_t W_{adj}DE_t^T))$

与之前的工作一致，将动态频域图视为隐藏扩散过程的转移矩阵，并结合前向和反向传播的预定义矩阵 $P$，以及 GWNet 中的自适应邻接矩阵 $A_{adp}$。于是得到如下的图卷积计算层：

$Z_t = \sum^K_{k=0}(P^kX_tW_{k,1} + A^k_{adp}X_tW_{k,2}+A_D^tX_tW_{k,3})$

其中，$K$ 表示图卷积邻域的阶数。

**五、代码理解：**

```python
# 2.construction of dynamic frequency domain graph
# =========================dynamic frequency domain graph===========================
# 2.1 FFC
xn1 = input[:, 0, :, -self.seq_len:]  # (B, N, T)
# Perform a 1D Fast Fourier Transform (FFT) on the last dimension
xn1 = torch.fft.rfft(xn1, dim=-1)  # (B, N, T // 2 + 1)
xn1 = torch.abs(xn1)
xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=1, eps=1e-12, out=None)
xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=2, eps=1e-12, out=None) * self.a
# 2.2 FC
xn1 = torch.matmul(xn1, self.Ex1)    # (B, N, fft_emb)
# Concat Traffic Freature, Identity Embedding and Time Labels
xn1k = self.cat(xn1, self.node_emb)  # (B, N, fft_emb + identity_emb)
T_D = self.T_i_D_emb[(data[:, :, :, 1] * 288).type(torch.LongTensor)][:, -1, :, :]
D_W = self.D_i_W_emb[(data[:, :, :, 2]).type(torch.LongTensor)][:, -1, :, :]
x_n1 = torch.cat([xn1k, T_D, D_W], dim=2)  # (B, N, fft_emb + identity_emb + seq_len * 2)

# 2.3 Conv1d
x1 = torch.bmm(x_n1.permute(1,0,2), self.Wd).permute(1, 0, 2)  # (B, N, hidden_emb)
x1 = torch.relu(x1)

# 2.4 Conv1d
x1k = self.layersnorm(x1)
x1k = self.drop(x1k)
adp = torch.einsum('bne, ek->bnk', x1k, self.Wxabs)

# 2.5 Transposition
adj = torch.bmm(adp, x1.permute(0, 2, 1))  # (B, N, N)
adp = torch.relu(adj)
adp = dy_mask_graph(adp, self.subgraph_size)
adp = F.softmax(adp, dim=2)
new_supports = new_supports + [adp]
# ===========================dynamic frequency domain graph==========================
```

按照论文模型图理解这段核心的代码，难点还是在于对 xn1 在最后一个维度上进行一维快速傅里叶变换 (FFT)。输入数据 xn1 的维度是 (B, N, T)，**torch.fft.rfft** 对实数输入进行 FFT，返回正频率部分。输出的维度将会是 (B, N, T//2+1)，因为实数输入的 rFFT 输出大小是输入大小的一半加一（正频率部分的长度）。

接着取绝对值可以得到频域中的幅度信息，这一步通常用于提取信号的频率特征。最后做了两次 L2 范数的归一化，第一步归一化的目的是让每个样本在特征维度上的幅度特征具有相同的尺度，第二步归一化标准化了时间维度的特征，使得每个时间步上的频率特征具有一致的尺度。





